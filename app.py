import streamlit as st
import numpy as np
import pulp
import pandas as pd
import time
import openpyxl

st.title('NSF Panel Assignment')
st.sidebar.title('Inputs')

rankings_sheet = st.sidebar.file_uploader("Upload an Excel file with rankings matrix", type = "xlsx")

num_proposals = int(st.sidebar.number_input('Number of proposals', min_value = 1, step = 1, format = "%d"))
num_reviewers = int(st.sidebar.number_input('Number of reviewers', min_value = 1, step = 1, format = "%d"))
reviews_per_proposal = int(st.sidebar.number_input('Number of reviews per proposal', min_value = 1, step = 1, format = "%d"))

if rankings_sheet is not None:
    rankings = pd.read_excel(rankings_sheet, header = None)
    rankings = rankings.to_numpy()

    if st.sidebar.button('Optimize'):

        start_time = time.time()

        total_reviews = num_proposals * reviews_per_proposal

        max_reviews_per_reviewer = int(np.ceil(total_reviews / num_reviewers))
        min_reviews_per_reviewer = int(np.floor(total_reviews / num_reviewers))
        extra_reviews = total_reviews % num_reviewers

        num_vars = num_reviewers * num_proposals

        f = rankings.T.flatten().reshape(-1, 1)
        
        prob = pulp.LpProblem("ReviewAssignment", pulp.LpMinimize)

        x = pulp.LpVariable.dicts('x', range(num_vars), lowBound = 0, upBound = 1, cat = 'Continuous')

        prob += pulp.lpSum([f[i] * x[i] for i in range(num_vars)])

        for j in range(num_proposals):
            prob += pulp.lpSum([x[i*num_proposals + j] for i in range(num_reviewers)]) == reviews_per_proposal

        for i in range(num_reviewers):
            prob += pulp.lpSum([x[i*num_proposals + j] for j in range(num_proposals)]) <= max_reviews_per_reviewer
            prob += pulp.lpSum([x[i*num_proposals + j] for j in range(num_proposals)]) >= min_reviews_per_reviewer

        for i in range(num_reviewers):
            for j in range(num_proposals):
                if rankings[i, j] == 0:
                    prob += x[i*num_proposals + j] == 0

        prob.solve()

        assignments = np.zeros((num_reviewers, num_proposals))
        for i in range(num_reviewers):
            for j in range(num_proposals):
                assignments[i, j] = pulp.value(x[i*num_proposals + j])

        fval = prob.objective.value()

        conflicts = (assignments.astype(bool) & (rankings == 0))
        if np.any(conflicts):
            st.error('Conflicts detected in the assignments')
            reviewer_idx, proposal_idx = np.where(conflicts)
            for k in range(len(reviewer_idx)):
                st.error(f'Reviewer {reviewer_idx[k]} assigned to Proposal {proposal_idx[k]} (conflict) \n')
        else:
            st.sidebar.success('No conflicts found in the review assignments.')

        fairness_metric = np.zeros((1, int(num_reviewers)))

        for reviewer in range(int(num_reviewers)):
            assigned_proposals = np.where(assignments[:, reviewer] == 1)[0]
            strong_preferences = np.sum(rankings[reviewer, assigned_proposals] == 1)
            medium_preferences = np.sum(rankings[reviewer, assigned_proposals] == 2)
            low_preferences = np.sum(rankings[reviewer, assigned_proposals] == 3)

            total_score = strong_preferences * 3 + medium_preferences * 2 + low_preferences * 1
            max_possible_score = len(assigned_proposals) * 3
            fairness_metric[0, reviewer] = total_score / max_possible_score * 100

        lead_assignments = np.zeros((num_proposals, int(num_reviewers)))
        lead_counts = np.zeros((1, int(num_reviewers)))
        proposals_assigned = np.zeros((num_proposals, 1), dtype = bool)

        total_preferences = np.sum(rankings, 1)
        rankings_nan = rankings.astype(float)
        rankings_nan[rankings_nan == 0] = np.nan
        average_preferences = np.nanmean(rankings_nan, 1)
        reviews_count = np.sum(assignments, 1)

        data = np.column_stack((reviews_count.flatten(), average_preferences))
        df = pd.DataFrame(data, columns=['reviews_count', 'average_preferences'])
        sorted_df = df.sort_values(by=['reviews_count', 'average_preferences'], ascending=[True, True])
        sorted_reviewers = sorted_df.index.values

        round_robin_index = 0

        for lead in range(num_proposals):
            reviewer = sorted_reviewers[round_robin_index]
            lead_counts[0, reviewer] += 1
            round_robin_index = (round_robin_index + 1) % int(num_reviewers)

        prob = pulp.LpProblem("LeadAssignment", pulp.LpMinimize)

        x = pulp.LpVariable.dicts('x', range(num_vars), lowBound=0, upBound=1, cat='Binary')

        prob += pulp.lpSum([f[i] * x[i] for i in range(num_vars)])

        for i in range(num_reviewers):
            prob += pulp.lpSum([x[i*num_proposals + j] for j in range(num_proposals)]) <= lead_counts.flatten()[i]

        for i in range(num_reviewers):
            prob += pulp.lpSum([x[i*num_proposals + j] for j in range(num_proposals)]) >= 1

        for i in range(num_vars):
            prob += x[i] <= assignments.flatten()[i]

        for j in range(num_proposals):
            prob += pulp.lpSum([x[i*num_proposals + j] for i in range(num_reviewers)]) == 1

        prob.solve()

        lead_assignments = np.zeros((num_reviewers, num_proposals))
        for i in range(num_reviewers):
            for j in range(num_proposals):
                lead_assignments[i, j] = pulp.value(x[i*num_proposals + j])

        lead_assignments = lead_assignments.T

        if np.any(np.sum(lead_assignments, 1) != 1):
            st.error('Error: Some proposals do not have exactly one lead.')
        
        for r in range(num_reviewers):
            for p in range(num_proposals):
                if lead_assignments[p, r] == 1 and assignments[r, p] == 0:
                    st.error(f'Error: Reviewer {r} is assigned as lead for proposal {p} but is not reviewing it.')

        fairness_lsr_metric = np.zeros((1, num_reviewers))

        for reviewer in range(num_reviewers):
            assigned_as_lsr = np.where(lead_assignments[:, reviewer] == 1)[0]
            strong_preferences = np.sum(rankings[reviewer, assigned_as_lsr] == 1)
            medium_preferences = np.sum(rankings[reviewer, assigned_as_lsr] == 2)
            low_preferences = np.sum(rankings[reviewer, assigned_as_lsr] == 3)

            total_score = strong_preferences * 3 + medium_preferences * 2 + low_preferences * 1
            max_possible_score = len(assigned_as_lsr) * 3
            fairness_lsr_metric[0, reviewer] = total_score / max_possible_score * 100

        conflict_found = False

        for proposal in range(num_proposals):
            for reviewer in range(num_reviewers):
                if lead_assignments[proposal, reviewer] == 1 and rankings[reviewer, proposal] == 0:
                    st.error(f'Conflict: Reviewer {reviewer} assigned to Proposal {proposal} despite conflict\n')
                    conflict_found = True  

        if not conflict_found:
            st.sidebar.success('No conflicts found in lead assignments.')

        combined_assignments = np.full((num_proposals, num_reviewers), '-', dtype = object)

        for proposal in range(num_proposals):
            for reviewer in range(num_reviewers):
                if lead_assignments[proposal, reviewer] == 1:
                    combined_assignments[proposal, reviewer] = 'LSR'
                elif assignments[reviewer, proposal] == 1:
                    combined_assignments[proposal, reviewer] = 'R'

        column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
        row_names = [f'Proposal {i+1}' for i in range(num_proposals)]
        combined_assignments_df = pd.DataFrame(combined_assignments, columns=column_names, index=row_names)
        st.subheader('Combined assignments:')
        st.table(combined_assignments_df)

        column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
        fairness_metric_df = pd.DataFrame(fairness_metric.reshape(1, -1), columns=column_names)
        st.subheader('Fairness metric:')
        st.table(fairness_metric_df.style.format("{:.2f}"))

        column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
        fairness_lsr_metric_df = pd.DataFrame(fairness_lsr_metric.reshape(1, -1), columns=column_names)
        st.subheader('Fairness LSR metric:')
        st.table(fairness_lsr_metric_df.style.format("{:.2f}"))

        column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
        reviews_result = reviews_count - lead_counts
        summary_df = pd.DataFrame([lead_counts.flatten(), reviews_result.flatten()], index=["Leads", "Reviews"], columns = column_names) 

        st.subheader('Summary:')
        st.table(summary_df.style.format("{:.0f}"))

        st.subheader('Fval:')
        st.write(f'The objective function value is {fval}.')

        end_time = time.time()

        st.subheader('Total simulation time:')
        st.write(f'The total run time is {round(end_time - start_time, 2)} seconds.')

else:
    if st.sidebar.button('Optimize'):
        st.error('The rankings matrix CSV file required to run the optimization is empty or has not been uploaded.')
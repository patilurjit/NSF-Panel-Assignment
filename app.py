import streamlit as st
import numpy as np
import pulp
import pandas as pd
import time
from io import BytesIO
import base64
import openpyxl

def generate_rankings_matrix_template(num_reviewers, num_proposals):
    rankings = np.zeros((num_proposals, num_reviewers))
    column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
    row_names = [f'Proposal {i+1}' for i in range(num_proposals)]
    rankings_sheet = pd.DataFrame(rankings, columns = column_names, index = row_names)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine = 'openpyxl') as writer:
        rankings_sheet.to_excel(writer, index = True, sheet_name = 'Rankings Matrix')

    buffer.seek(0)
    return buffer

def generate_rankings(num_reviewers, num_proposals):
    rankings = np.zeros((num_reviewers, num_proposals), dtype=int)
    
    for i in range(num_reviewers):
        ranking = np.random.randint(1, 4, size=num_proposals)
        num_zeros = np.random.randint(0, 3)
        zero_positions = np.random.choice(num_proposals, num_zeros, replace=False)
        ranking[zero_positions] = 0
        rankings[i, :] = ranking
    return rankings

st.title('NSF Panel Assignment')
st.sidebar.title('Inputs')

method = st.radio("Method of optimization", ['LSR', 'LS_diff'])

st.markdown("""
    <style>
    .full-width-table {
        width: 80%;
        overflow-x: auto;
        overflow-y: hidden;
    }
    .full-width-table table {
        width: 80%;
        table-layout: fixed;
    }
    .full-width-table th, .full-width-table td {
        text-align: center;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 14px; /* Adjust the font size here */
    }
    .full-width-table th {
        min-width: 100px; /* Adjust this value to ensure header texts fit well */
        font-size: 16px; /* Adjust the font size for headers here */
    }
    .css-13sdm1b.e16nr0p33 {
      margin-top: -75px;
    }
    </style>
    """, unsafe_allow_html=True)

# number_input inputs (variable)
num_proposals = int(st.sidebar.number_input('Number of proposals', min_value = 1, step = 1, format = "%d"))
reviews_per_proposal = int(st.sidebar.number_input('Number of reviews per proposal', min_value = 1, step = 1, format = "%d"))
max_reviews_per_reviewer = int(st.sidebar.number_input('Maximum number of reviews per reviewer', min_value = 1, step = 1, format = "%d"))

# number_input inputs (fixed)
total_reviews = num_proposals * reviews_per_proposal
min_reviwers_val = int(np.ceil(total_reviews / max_reviews_per_reviewer))
max_reviewers_val = total_reviews
st.sidebar.markdown(f'Number of reviewers should be between {min_reviwers_val} and {max_reviewers_val}')
num_reviewers = int(st.sidebar.number_input('Number of reviewers', min_value = min_reviwers_val, max_value = max_reviewers_val, step = 1, format = "%d"))

# generate rankings matrix template
if st.sidebar.button('Generate rankings matrix template'):
    buffer = generate_rankings_matrix_template(num_reviewers, num_proposals)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="rankings_matrix_template.xlsx">Download the rankings matrix template</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# rankings matrix input
rankings_sheet = st.sidebar.file_uploader("Upload the filled rankings matrix template", type = "xlsx")

if rankings_sheet is not None:
    rankings = pd.read_excel(rankings_sheet, index_col = 0)
    if np.any(rankings != 0):
        rankings = rankings.to_numpy()
        rankings = rankings.T

        if st.sidebar.button('Optimize'):
            if method == 'LSR':

                start_time = time.time()

                min_reviews_per_reviewer = int(np.floor(total_reviews / num_reviewers))
                max_reviews_per_reviewer = int(np.ceil(total_reviews / num_reviewers))
                extra_reviews = total_reviews % num_reviewers

                num_vars = num_reviewers * num_proposals

                f = rankings.flatten().reshape(-1, 1)
                
                num_vars = num_reviewers * num_proposals

                prob = pulp.LpProblem("ReviewAssignment", pulp.LpMinimize)

                x = pulp.LpVariable.dicts('x', range(num_vars), lowBound = 0, upBound = 1, cat = 'Binary')

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

                lead_assignments = np.zeros((num_proposals, reviews_per_proposal))
                lead_counts = np.zeros(num_reviewers)
                proposals_assigned = np.zeros(num_proposals, dtype = bool)

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
                    lead_counts[reviewer] += 1
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

                lead_fval = prob.objective.value()

                conflicts = (assignments.astype(bool) & (rankings == 0))
                if np.any(conflicts):
                    st.error('Conflicts detected in the assignments:')
                    reviewer_idx, proposal_idx = np.where(conflicts)
                    for k in range(len(reviewer_idx)):
                        st.error(f'Reviewer {reviewer_idx[k]} assigned to Proposal {proposal_idx[k]} (conflict) \n')
                else:
                    st.sidebar.success('No conflicts found in the review assignments.')

                conflicts = (lead_assignments.T.astype(bool) & (rankings == 0))
                if np.any(conflicts):
                    st.error('Conflicts detected in the lead assignments:')
                    reviewer_idx, proposal_idx = np.where(conflicts)
                    for k in range(len(reviewer_idx)):
                        st.error(f'Reviewer {reviewer_idx[k]} assigned to Proposal {proposal_idx[k]} (conflict) \n')
                else:
                    st.sidebar.success('No conflicts found in the lead assignments.')

                proposal_count = np.sum(assignments, 0)

                for proposal in range(num_proposals):
                    if proposal_count[proposal] != reviews_per_proposal:
                        st.error(f'Proposal {proposal} does not have the required number of reviews.')
                    
                lead_proposal_count = np.sum(lead_assignments, 1)

                for proposal in range(num_proposals):
                    if lead_proposal_count[proposal] != 1:
                        st.error(f'Proposal {proposal} does not have the required number of leads.')
                    
                combined_assignments = np.full((num_proposals, num_reviewers), '-', dtype = object)
                fval_assignments = np.zeros((num_proposals, num_reviewers))

                for proposal in range(num_proposals):
                    for reviewer in range(num_reviewers):
                        if lead_assignments[proposal, reviewer] == 1:
                            combined_assignments[proposal, reviewer] = 'LSR'
                            fval_assignments[proposal, reviewer] = 2+3+1
                        elif assignments[reviewer, proposal] == 1:
                            combined_assignments[proposal, reviewer] = 'R'
                            fval_assignments[proposal, reviewer] = 1
                        elif rankings[reviewer, proposal] == 0:
                            combined_assignments[proposal, reviewer] = 'COI'
                        else:
                            combined_assignments[proposal, reviewer] = '-'

                fval_combined = np.sum(rankings.T * fval_assignments)

                proposal_count = np.sum(assignments, 0)
                fairness_prop_count = np.zeros(num_proposals)

                for proposal in range(num_proposals):
                    fairness_prop_count[proposal] = np.sum(assignments[:, proposal]*rankings[:, proposal])/proposal_count[proposal]

                reviews_count = np.sum(assignments, 1)
                fairness_metric = np.zeros(num_reviewers)

                for reviewer in range(num_reviewers):
                    fairness_metric[reviewer] = np.sum(assignments[reviewer, :] * rankings[reviewer, :]) / reviews_count[reviewer]

                fairness_lsr_metric = np.zeros(num_reviewers)

                for reviewer in range(num_reviewers):
                    fairness_lsr_metric[reviewer] = np.sum(lead_assignments[:, reviewer] * rankings[reviewer, :]) / lead_counts[reviewer]

                num_conflicts_proposal = np.sum(rankings == 0, 0)
                proposal_indices = list(range(num_proposals))

                con_reordered_combined_assignments = np.full((num_proposals, num_reviewers), '-', dtype = object)
                reviewer_conflict_matrix = (rankings == 0)

                grouped_proposals = []
                for reviewer in range(num_reviewers):
                    conflicts = np.where(reviewer_conflict_matrix[reviewer, :])[0].tolist()
                    grouped_proposals.extend(conflicts)

                grouped_proposals = list(dict.fromkeys(grouped_proposals))
                remaining_proposals = [item for item in proposal_indices if item not in grouped_proposals]
                grouped_proposals.extend(remaining_proposals)

                for i in range(num_proposals):
                    proposal = grouped_proposals[i]
                    for reviewer in range(num_reviewers):
                        if lead_assignments[proposal, reviewer] == 1:
                            con_reordered_combined_assignments[i, reviewer] = 'LSR'
                        elif assignments[reviewer, proposal] == 1:
                            con_reordered_combined_assignments[i, reviewer] = 'R'
                        elif rankings[reviewer, proposal] == 0:
                            con_reordered_combined_assignments[i, reviewer] = 'COI'
                        else:
                            con_reordered_combined_assignments[i, reviewer] = '-'

                column_names = [f'Reviewer {i}' for i in range(1, num_reviewers + 1)]
                row_names = [f'Proposal {i+1}' for i in grouped_proposals]
                con_reordered_combined_assignments_table = pd.DataFrame(con_reordered_combined_assignments, columns=column_names, index=row_names)

                num_proposals = lead_assignments.shape[0]
                num_reviewers = lead_assignments.shape[1]

                total_leads = np.sum(lead_assignments, 0)

                proposal_discussion_order = []

                discussed_proposals = np.zeros(num_proposals, dtype = bool)

                sorted_reviewer_indices = np.argsort(-total_leads)

                for reviewer_index in range(num_reviewers):
                    reviewer = sorted_reviewer_indices[reviewer_index]
                    reviewer_leads = np.where(lead_assignments[:, reviewer] == 1)[0]
                    for proposal in reviewer_leads:
                        if not discussed_proposals[proposal]:
                            proposal_discussion_order.append(proposal)
                            discussed_proposals[proposal] = True

                sorted_combined_assignments = np.full((num_proposals, num_reviewers), '-', dtype = object)

                for i in range(num_proposals):
                    original_proposal = proposal_discussion_order[i]
                    sorted_combined_assignments[i, :] = combined_assignments[original_proposal, :]

                column_names = [f'Reviewer {i}' for i in range(1, num_reviewers + 1)]
                row_names = [f'Proposal {i+1}' for i in proposal_discussion_order]
                sorted_combined_assignments_table = pd.DataFrame(sorted_combined_assignments, columns = column_names, index = row_names)

                ratings = ['E', 'V', 'G', 'F', 'P', 'E/V', 'V/G', 'G/F', 'F/P']

                num_proposals = combined_assignments.shape[0]
                num_reviewers = combined_assignments.shape[1]

                ratings_matrix = np.full((num_proposals, num_reviewers), '-', dtype = object)

                for proposal in range(num_proposals):
                    for reviewer in range(num_reviewers):
                        if assignments[reviewer, proposal] != 0:
                            random_rating = np.random.choice(ratings)
                            ratings_matrix[proposal, reviewer] = random_rating
                        else:
                            ratings_matrix[proposal, reviewer] = '-'

                column_names = [f'Reviewer {i}' for i in range(1, num_reviewers + 1)]
                row_names = [f'Proposal {i+1}' for i in range(num_proposals)]

                ratings_table = pd.DataFrame(ratings_matrix, columns = column_names, index = row_names)

                ratings_matrix = ratings_table.to_numpy(dtype = str)

                rating_to_points = {
                        'E': 1, 'V': 2, 'G': 3, 'F': 4, 'P': 5,
                        'E/V': (1 + 2) / 2, 'V/G': (2 + 3) / 2, 'G/F': (3 + 4) / 2, 'F/P': (4 + 5) / 2
                    }

                points_matrix = np.empty((num_proposals, num_reviewers), dtype = object)

                for proposal in range(num_proposals):
                        for reviewer in range(num_reviewers):
                            rating = ratings_matrix[proposal, reviewer]
                            if rating:
                                points_matrix[proposal, reviewer] = str(rating_to_points.get(rating, ''))
                            else:
                                points_matrix[proposal, reviewer] = ''

                numeric_points_matrix = np.full((num_proposals, num_reviewers), np.nan)
                for proposal in range(num_proposals):
                    for reviewer in range(num_reviewers):
                        if points_matrix[proposal, reviewer]:
                            numeric_points_matrix[proposal, reviewer] = float(points_matrix[proposal, reviewer])

                total_points_per_proposal = np.nansum(numeric_points_matrix, axis = 1)
                sorted_indices = np.argsort(total_points_per_proposal)
                sorted_combined_assignments = combined_assignments[sorted_indices, :]
                total_points_for_sorted_proposals = total_points_per_proposal[sorted_indices]

                new_row_names = [f'Proposal {i+1}' for i in sorted_indices]

                sorted_combined_assignments_df = pd.DataFrame(sorted_combined_assignments, columns=column_names, index = new_row_names)
                total_points_df = pd.DataFrame(total_points_for_sorted_proposals, columns = ['Total Points'], index = new_row_names)

                rating_combined_assignments_with_total = pd.concat([sorted_combined_assignments_df, total_points_df], axis = 1)

                column_names = [f'Proposal {i}' for i in range(1, num_proposals + 1)]
                fairness_prop_count_df = pd.DataFrame(fairness_prop_count.reshape(1, -1), index = ['Value'], columns = column_names)

                column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
                fairness_metric_df = pd.DataFrame(fairness_metric.reshape(1, -1), index = ['Value'], columns = column_names)

                column_names = [f'Reviewer {i+1}' for i in range(num_reviewers)]
                fairness_lsr_metric_df = pd.DataFrame(fairness_lsr_metric.reshape(1, -1), index = ['Value'], columns=column_names)

                st.subheader('Conflicts Reordered combined assignment matrix (Proposals x Reviewers):')
                st.write(con_reordered_combined_assignments_table.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('Leads Reordered combined assignment matrix (Proposals x Reviewers):')
                st.write(sorted_combined_assignments_table.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('Rating Reordered combined assignment matrix (Proposals x Reviewers):')
                st.write(rating_combined_assignments_with_total.to_html(classes='full-width-table'), unsafe_allow_html=True)

                combined_assignments_df = pd.DataFrame(combined_assignments, columns = column_names, index = row_names)
                st.subheader('Combined assignment matrix (Proposals x Reviewers):')
                st.write(combined_assignments_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('Fairness metric for each reviewer:')
                st.write(fairness_metric_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('Fairness metric for each LSR:')
                st.write(fairness_lsr_metric_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                reviews_count_df = pd.DataFrame(reviews_count.reshape(1, -1), index = ['Value'], columns = [f'Reviewer {i + 1}' for i in range(num_reviewers)])
                st.subheader('Number of total reviews per reviewer:')
                st.write(reviews_count_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                lead_counts_df = pd.DataFrame(lead_counts.reshape(1, -1), index = ['Value'], columns = [f'Reviewer {i + 1}' for i in range(num_reviewers)])
                st.subheader('Number of leads per reviewer:')
                st.write(lead_counts_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                proposal_count_df = pd.DataFrame(proposal_count.reshape(1, -1), index = ['Value'], columns = row_names)
                st.subheader('Number of reviews per proposal:')
                st.write(proposal_count_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                lead_proposal_count_df = pd.DataFrame(lead_proposal_count.reshape(1, -1), index = ['Value'], columns = row_names)
                st.subheader('Number of leads per proposal:')
                st.write(lead_proposal_count_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('Fairness metric per proposal:')
                st.write(fairness_prop_count_df.to_html(classes='full-width-table'), unsafe_allow_html=True)

                st.subheader('fval:')
                st.write(f'The objective function value is {fval}.')

                st.subheader('fval_combined:')
                st.write(f'The combined objective function value is {fval_combined}.')

                end_time = time.time()

                st.subheader('Total simulation time:')
                st.write(f'The total run time is {round(end_time - start_time, 2)} seconds.')
                
                def create_excel_from_dataframe(dataframes_with_sheets):
                    workbook = openpyxl.Workbook()
                    default_sheet = workbook.active
                    default_sheet.title = 'Default Sheet'
                    
                    workbook.remove(default_sheet)
                    
                    for dataframe, sheet_name in dataframes_with_sheets:
                        worksheet = workbook.create_sheet(title=sheet_name)
                        
                        with BytesIO() as temp_buffer:
                            with pd.ExcelWriter(temp_buffer, engine='openpyxl') as writer:
                                dataframe.to_excel(writer, index=True, sheet_name=sheet_name)
                            
                            temp_buffer.seek(0)
                            temp_workbook = openpyxl.load_workbook(temp_buffer)
                            temp_sheet = temp_workbook[sheet_name]
                            
                            for row in temp_sheet.iter_rows(values_only=True):
                                worksheet.append(row)
                    
                    output = BytesIO()
                    workbook.save(output)
                    output.seek(0)
                    
                    return output

                dataframes_with_sheets = [
                    (con_reordered_combined_assignments_table, 'Conflicts Reordered combined assignment matrix'),
                    (sorted_combined_assignments_table, 'Leads Reordered combined assignment matrix'),
                    (rating_combined_assignments_with_total, 'Rating Reordered combined assignment matrix'),
                    (combined_assignments_df, 'Combined assignment matrix'),
                    (fairness_metric_df, 'Fairness metric'),
                    (fairness_lsr_metric_df, 'Fairness LSR metric'),
                    (reviews_count_df, 'Number of total reviews per reviewer'),
                    (lead_counts_df, 'Number of leads per reviewer'),
                    (proposal_count_df, 'Number of reviews per proposal'),
                    (lead_proposal_count_df, 'Number of leads per proposal'),
                    (fairness_prop_count_df, 'Fairness metric per proposal')
                ]

                excel_buffer = create_excel_from_dataframe(dataframes_with_sheets)

                if excel_buffer is not None:
                    st.download_button(
                        label="Download Excel file",
                        data=excel_buffer,
                        file_name="dataframe.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            elif method == 'LS_diff':
                pass


    else:
        st.error('The Excel file with the rankings matrix required to run the optimization is empty.')

else:
    if st.sidebar.button('Optimize'):
        st.error('The Excel file with the rankings matrix required to run the optimization has not been uploaded.')
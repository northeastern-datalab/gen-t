This folder contains code to run the Discovery and Integration phases of Gen-T, and baseline ALITE and ALITE-PS.

## Gen-T Codes
1. (Optional) In our experiments, we ran an existing, state-of-the-art table disovery system that retrieves a set of candidate tables from a data lake in an efficient manner: Starmie (https://github.com/megagonlabs/starmie)
2. In folder (findCandidate/), set_similarity.py finds a set of candidate tables either from step 1 or from the data lake. To do so, it finds tables containing columns with high set overlap with columns in the Source Table.
3. In folder (discovery/), recover_matrix_ternary.py finds a set of originating tables from the set of candidate tables. To do so, it creates matrix representations of each table containing values -1, 0, 1 to encode if each value in each aligned column in each aligned tuple with the Source Table is non-null and not matching, null when the Source Table's value is non-null, and matching the Source Table's value, respectively. Next, it combines the matrix representations to ideally produce a matrix containing all 1's. Finally, it returns the tables whose matrices were used in the matrix integrations as the set of originating tables.
4. In folder (integration/), table_integration.py combines the set of originating tables using Outer Union, Selection, Projection, Complementation, and/or Subsumption operators. 

## Baseline: ALITE and ALITE-PS
In folder (integration/), alite_fd_original.py is the code used to run ALITE and ALITE-PS. They are given the same set of candidate tables used in Gen-T (note this is not the set of originating tables).


## Reproducibility:
1. To get the set of candidate tables with high set similarity and the set of originating tables found in Gen-T, run discovery/runFindPath.py. This saves both sets of tables in a folder called "results_candidate_tables".
2. To run table integration for Gen-T, run integration/runIntegration.py and set --genT to 1.
3. To run ALITE, also run integration/runIntegration.py. Ensure that --doPS is set to 0.
4. To run ALITE-PS, also run integration/runIntegration.py. Ensure that --doPS is set to 1.
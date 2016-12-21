__author__ = 'kazjon'
import csv

dois = []
i_path = "ACMDL/Pique_Research papers_v1.0.csv"
o_path = "ACMDL/Pique_Research papers_v1.0_pruned.csv"
o_rows = []

with open(i_path,"rb") as i_f:
	for row in csv.DictReader(i_f):
		if not row["DOI"] in dois:
			o_rows.append(row)
			dois.append(row["DOI"])

with open(o_path,"wb") as o_f:
	w = csv.DictWriter(o_f,fieldnames=o_rows[0].keys())
	w.writeheader()
	w.writerows(o_rows)

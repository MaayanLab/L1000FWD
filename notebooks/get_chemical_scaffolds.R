library(rcdk)

drugs_df <- read.csv('drugs_smiles.csv', stringsAsFactors = F)

smiles2scaffold <- function(smiles){
  mol = parse.smiles(smiles)[[1]]
  frags = get.murcko.fragments(mol)
  frags
}


N = dim(drugs_df)[1]
all_rings = vector(mode = 'character', length = N)
all_scaffolds = vector(mode = 'character', length = N)

# for (i in 1:N){
# for (i in 5101:N){
for (i in 19166:N){
  smiles = drugs_df[i, 'canonical_smiles']
  frags = smiles2scaffold(smiles)
  rings = paste(frags[[1]]$rings, collapse = ';')
  scaffolds = paste(frags[[1]]$frameworks, collapse = ';')
  
  all_rings[i] = rings
  all_scaffolds[i] = scaffolds
}
drugs_df$rings = all_rings
drugs_df$scaffolds = all_scaffolds

write.csv(drugs_df, file = 'drugs_smiles_rings_scaffolds.csv', row.names = F)

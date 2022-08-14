# the libraries we're using
library(factoextra)
library(cluster)

"------------------------------------------------------------------------------"

# load and prep data
#load data
df <- USArrests
#remove rows with missing values
df <- na.omit(df)
#scale each variable to have a mean of 0 and sd of 1
df <- scale(df)
#view first six rows of dataset
head(df)

"------------------------------------------------------------------------------"

# Find the optimal number of clusters
fviz_nbclust(df, kmeans, method = "wss")

# Another form is the Gap Statistic 
#calculate gap statistic based on number of clusters
"variación intra-clúster total para diferentes valores de k con sus valores 
esperados para una distribución sin agrupación"
gap_stat <- clusGap(df,
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 10,
                    B = 50)

#plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

"------------------------------------------------------------------------------"

# Perform K-means Clustering with optimal k
#make this example reproducible
set.seed(1)

#perform k-means clustering with k = 4 clusters
km <- kmeans(df, centers = 4, nstart = 25)

#view results
km


# 13 states were assigned to the first cluster
# 13 states were assigned to the second cluster
# 16 states were assigned to the third cluster
# 8 states were assigned to the fourth cluster

"------------------------------------------------------------------------------"

#plot results of final k-means model
fviz_cluster(km, data = df)

"------------------------------------------------------------------------------"

#find means of each cluster
aggregate(USArrests, by=list(cluster=km$cluster), mean)


# We interpret this output is as follows:
  
# The mean number of murders per 100,000 citizens among the states in cluster 1 is 3.6.
# The mean number of assaults per 100,000 citizens among the states in cluster 1 is 78.5.
# The mean percentage of residents living in an urban area among the states in cluster 1 is 52.1%.
# The mean number of rapes per 100,000 citizens among the states in cluster 1 is 12.2.

"------------------------------------------------------------------------------"

# add cluster assigment to original data
final_data <- cbind(USArrests, cluster = km$cluster)

# view final data
head(final_data)

"------------------------------------------------------------------------------"
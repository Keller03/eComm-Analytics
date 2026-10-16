segment_features = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership",
    "Yearly Amount Spent",
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[segment_features])

kmeans = KMeans(n_clusters=3, random_state=69, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

cluster_profiles = df.groupby("Cluster")[segment_features].mean()
clusters_by_spend = cluster_profiles["Yearly Amount Spent"].sort_values().index
segment_names = ["at_risk", "mid", "high_value"]
segment_by_cluster = dict(zip(clusters_by_spend, segment_names))
df["Segment"] = df["Cluster"].map(segment_by_cluster)

segment_profiles = (
    df.groupby("Segment")[segment_features]
    .agg(["count", "mean"])
    .reindex(segment_names)
)
print("\nKMeans-Segmente (at_risk ist ein Proxy, keine Churn-Vorhersage):")
print(segment_profiles.round(2))

output_path = Path(__file__).with_name("customer_segments.csv")
df[["Email", "Segment"]].to_csv(output_path, index=False)
print(f"\nKundensegmente gespeichert: {output_path}")
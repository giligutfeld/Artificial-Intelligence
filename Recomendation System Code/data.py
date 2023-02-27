# Gili Gutfeld 209284512

def watch_data_info(data):

    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):

    sorted_ratings = data['ProductId'].value_counts(sort=True)
    sorted_users = data['UserId'].value_counts(sort=True)

    print(f"number of users are :  {len(sorted_users)}")
    print(f"number of products ranked are : {len(sorted_ratings)}")
    print(f"number of ranking are: {data['Rating'].count()}")
    print(f"minimum number of ratings given to a product : {sorted_ratings[-1]}")
    print(f"maximum number of ratings given to a product : {sorted_ratings[0]}")
    print(f"minimum number of products ratings by user : {sorted_users[-1]}")
    print(f"maximum number of products ratings by user : {sorted_users[0]}")

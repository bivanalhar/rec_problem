from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')

algorithm = SVD()

cross_validate(algorithm, data, measures = ['RMSE', 'MAE'], cv = 5, verbose = True)
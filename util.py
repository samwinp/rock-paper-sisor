# import json
# import pickle
# import numpy as np

# __location = None
# __data_colums = None
# __model = None

# def load_saved_data():
#     global __location, __data_colums, __model

#     with open('./columns.json', 'r') as f:
#         __data_colums = json.load(f)

#     with open('./house.pickle', 'rb') as f:
#         __model = pickle.load(f)

# # Call the load function to populate __data_colums and __model when util is imported
# load_saved_data()

# def get_location_names():
#     return __data_colums

# def get_estimated_price(location, total_sqft, bath, BHK):
#     loc_index = __data_colums['data_columns'].index(location)
#     x = np.zeros(len(__data_colums['data_columns']))   
#     x[0] = total_sqft
#     x[1] = bath
#     x[2] = BHK
#     if loc_index > 0:
#         x[loc_index] = 1
#     price = __model.predict([x])[0]
#     return int(price)

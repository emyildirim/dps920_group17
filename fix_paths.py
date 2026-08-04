
## Fix Paths
# udacity self driving car simulator on mac used as a simulator
# the image paths recorded in driving_log.csv uses full path, meaning it will be broken 
# and unusable upon changing the directory or sharing with someone else.
# Later changed the 'driving_log_cleaned.csv" to "driving_log.csv"

# NOTE: This file is used ONCE and wont be used again unless data samples are recollected through simulator again.

import pandas as pd

columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
data = pd.read_csv('driving_log.csv', names=columns)

def make_relative_path(absolute_path):
    filename = absolute_path.replace('\\', '/').split('/')[-1]
    
    #returns relative path
    return 'IMG/' + filename

data['Center'] = data['Center'].apply(make_relative_path)
data['Left']   = data['Left'].apply(make_relative_path)
data['Right']  = data['Right'].apply(make_relative_path)

#saves the cleaned data to a new CSV file
data.to_csv('driving_log_cleaned.csv', index=False, header=False)
print("Path cleaning done!")
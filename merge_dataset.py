import pandas as pd 
import os

## -------------------- Update Dataframe function --------------- ##
def update_df(item, id_):
    global df

    vid_name = os.path.join(game_path, id_, item[:-9]+'.mp4')
    csv_name = os.path.join(game_path, id_, item)

    # Read .CSV file
    df_ball = pd.read_csv(csv_name)
    
    for row_idx in range(len(df_ball)-2):
        row1 = df_ball.iloc[row_idx]
        row2 = df_ball.iloc[row_idx+1]
        row3 = df_ball.iloc[row_idx+2]

        df.loc[len(df)] = [vid_name, row_idx, row1.X ,row1.Y, row2.X, row2.Y, row3.X, row3.Y]

    
## -------------------------------------------------------------- ##


# games path
game_path = 'games'

# find total game id in that folder
game_ids = os.listdir(game_path)

# Result dataframe
df = pd.DataFrame(columns = ['video_path', 'frame_idx', 'cord_1_x', 'cord_1_y', 'cord_2_x', 'cord_2_y', 'cord_3_x', 'cord_3_y'])

# Iterate on all ids
for id_ in game_ids:
    print(id_)
    # Get all files, included .csv, .mp4, .pkl
    files_list = os.listdir(os.path.join(game_path,id_))
    for item in files_list:
        if item[-3:] == 'csv':
            update_df(item, id_)


print(df)
df.to_csv('merged_dataset.csv')
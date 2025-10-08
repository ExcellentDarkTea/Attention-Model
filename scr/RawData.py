import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

def flatten_signal(signal):
    """Flatten nested lists/arrays to 1D."""
    arr = np.array(signal)
    return arr.flatten()

def process_all_signals(df_stress, df_time):
    result = []
    for participant in df_stress.index:
        for signal_type in df_stress.columns:
            if signal_type == 'tags':
                continue  # skip tags
            signal = np.array(df_stress.loc[participant, signal_type])
            time = df_time.loc[participant, signal_type]
            time_index = pd.to_datetime(time)
            if signal_type == 'ACC':
                # ACC is Nx3, so create columns for X, Y, Z
                df = pd.DataFrame(signal, columns=['X', 'Y', 'Z'], index=time_index)
                df = df.resample('1S').mean().interpolate()
                df.reset_index(inplace=True)
                df['participant'] = participant
                df['signal_type'] = signal_type
                df['time'] = df['index']
                result.append(df[['time', 'X', 'Y', 'Z', 'participant', 'signal_type']])
            else:
                # 1D signals
                signal = signal.flatten()
                df = pd.DataFrame({signal_type: signal}, index=time_index)
                df = df.resample('1S').mean().interpolate()
                df.reset_index(inplace=True)
                df['participant'] = participant
                df['signal_type'] = signal_type
                df = df.rename(columns={'index': 'time', signal_type: 'signal'})
                result.append(df[['time', 'signal', 'participant', 'signal_type']])
    return pd.concat(result, ignore_index=True)

def create_df_array(dataframe):
    matrix_df=dataframe.values
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()
    return array_df

# convert UTC arrays to arrays in seconds
def time_abs_(UTC_array):
    new_array=[]
    for utc in UTC_array:
        time=(datetime.datetime.strptime(utc,'%Y-%m-%d %H:%M:%S')-datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds()
        new_array.append(int(time))
    return new_array

def read_signals(main_folder):
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    subfolders = next(os.walk(main_folder))[1]
    utc_start_dict={}
    for folder_name in subfolders:
            csv_path = f'{main_folder}/{folder_name}/EDA.csv'
            df=pd.read_csv(csv_path)
            utc_start_dict[folder_name]= df.columns.tolist()

    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        # Get a list of files
        files = os.listdir(folder_path)
        signals = {}
        time_line = {}
        fs_signal= {}
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv','tags.csv','ACC.csv']
   

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path,header=None)
                        tags_vector = create_df_array(df)
                        tags_UTC_vector =np.insert(tags_vector,0,utc_start_dict[folder_name])
                        signal_array=time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array=[]
                
                else:
                    df = pd.read_csv(file_path)
                    fs= df.loc[0]
                    # print(f"Sampling frequency for {file_name} in {folder_name}: {fs}")
                    fs=int(fs[0])
                    df.drop([0],axis = 0,inplace=True) 
                    signal_array = df.values
                    time_array = np.linspace(0, len(signal_array)/fs,len(signal_array))
                    #add time from title of df
                    utc_start = utc_start_dict[folder_name][0]
                    # print(f"UTC start for {file_name} in {folder_name}: {utc_start}")

                    time_array = [datetime.datetime.strptime(utc_start, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=t) for t in time_array]

                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array

                fs_signal[signal_name] = fs

        # Store the signals of the current subfolder in the main dictionary
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal

    return signal_dict, time_dict, fs_dict

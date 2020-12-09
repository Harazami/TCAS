import streamlit as st
import pandas as ps
import pickle

st.write(""" 

## My First Web Application 
Let's fill your information **TCAS**! 

""")

st.sidebar.header('Student Input')
st.sidebar.subheader('Please fill your data:')

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    v_AcademicYear = st.sidebar.radio('Academic Year', ['2562','2563'])
    v_AcademicSemester = st.sidebar.radio('Academic Semester', ['1', '2', '3', '4'])
    v_FacultyID = st.sidebar.text_input('Faculty ID')
    v_DepartmentCode = st.sidebar.text_input('Department Code')
    v_EntryTryID = st.sidebar.text_input('Entry Try ID')
    v_EntryGroupID = st.sidebar.text_input('Entry Group ID')
    v_TCAS = st.sidebar.radio('TCAS rounds', ['1', '2', '3', '4', '5'])
    v_LevelID = st.sidebar.radio('Level ID',['1', '2', '3', '4'])
    v_StudentTH = st.sidebar.radio('Student TH', ['Foreign', 'Thai'])


   # Change the value of sex to be {'F', 'T'} as stored in the trained dataset
    if v_StudentTH == 'Foreign':
        v_StudentTH = '1'
    else :
        v_StudentTH = '0'

    # Store user input data in a dictionary
    data = {'Year': v_AcademicYear,
            'Academic Semester': v_AcademicSemester,
            'Faculty_ID': v_FacultyID,
            'Department_Code': v_DepartmentCode,
            'Entry_Try_ID': v_EntryTryID,
            'Entry_Group_ID': v_EntryGroupID,
            'TCAS': v_TCAS,
            'Level_ID': v_LevelID,
            'Nationailty_Student': v_StudentTH}

    # Create a data frame from the above dictionary
    data_df = ps.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application for MFU TCAS student check:')
st.write(""" Please **insert your Data** at sidebar before check the data!! """)
st.write("""If not, the prediction will be on **error** """)

# -- Display new data from user inputs:
st.subheader('Student Information:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = ps.read_csv('data_knn.csv')
df = ps.concat([df, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = ps.get_dummies(df[['StudentType']])

#Combine all transformed features together
X_new = ps.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

X_new = X_new.drop(columns=['StudentType','AcademicYear', 'AcademicSemester','Sex','FacultyID','DepartmentCode','EntryTypeID','EntryGroupID','LevelID','StudentType','Status'])


# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)
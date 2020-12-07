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
data_sample = ps.read_excel('tcas.xlsx')
df = ps.concat([df, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = ps.get_dummies(df[['StudentType']])

#Combine all transformed features together
X = ps.concat([cat_data, df], axis=1)
X = X[:1] # Select only the first row (the user input data)

#Drop un-used feature
X = X.drop(columns=['StudentType','Sex','AcademicYear','AcademicSemester','PrefixName','FacultyID','FacultyName','DepartmentCode','DepartmentName','MajorName',
'EntryTypeID','EntryTypeName','EntryGroupID','EntryGroupName','LevelID','LevelName','LevelNameEng','ApplicationDate','EntryGPA','HomeRegion','StudentTH',
'Country','SchoolName','SchoolProvince','SchoolRegion','SchoolRegionName','SchoolRegionNameEng','NationName','ReligionName','ProvinceNameEng','GPAX','GPA_Eng','GPA_Math','GPA_Sci','GPA_Sco',
'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27',
'Q28','Q29','Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q41','Q42','Status'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X)

# -- Reads the saved normalization model
load_sc = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X = load_sc.transform(X)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)
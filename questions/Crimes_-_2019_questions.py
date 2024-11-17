data = [
    {
        "question": "How many columns does the dataset contain and how many columns contain null values? Print the result as 'Number of Columns' and 'Number of Columns with Null Value'.",
        "ground_truth": "Number of Columns: 22, Number of Columns with Null Value: 7 ",
        "difficulty" : "easy",
        "type" : "data quality assessment"
    },
    {
        "question": "Check if there are any duplicate rows and count them. Print the result as 'Number of Duplicates:' and 'Integer'.",
        "ground_truth": "Number of Duplicates: 0",
        "difficulty" : "easy",
        "type" : "data quality assessment"
    },
    {
        "question": "Find all the columns that only contain the same value in each row. Print the result as 'Columns with the same value in all rows:' : 'Column Name' or None if no such column exists.",
        "ground_truth": "Columns with the same value in all rows: 'Year'",
        "difficulty" : "easy",
        "type" : "data quality assessment"
    },
    {
        "question": "Name all columns that contain null values and calculate the percentage of null values to non null values rounded to three values after the comma. Print the result as 'Column Name' : 'Percentage'.",
        "ground_truth": "'Location Description' : 0.509%, 'Ward' : 0.006%, 'X Coordinate' : 0.899%, 'Y Coordinate' : 0.899%, 'Latitude' : 0.899%, 'Longitude' : 0.899%, 'Location' : 0.899%",
        "difficulty" : "medium",
        "type" : "data quality assessment"
    },
    {
        "question": "Print all values that occur in the column about crime types.",
        "ground_truth": "'ROBBERY' 'OFFENSE INVOLVING CHILDREN' 'THEFT' 'SEX OFFENSE' 'CRIMINAL SEXUAL ASSAULT' 'BURGLARY' 'HOMICIDE' 'DECEPTIVE PRACTICE' 'BATTERY' 'OTHER OFFENSE' 'MOTOR VEHICLE THEFT' 'WEAPONS VIOLATION' 'STALKING' 'NARCOTICS' 'ASSAULT' 'CRIMINAL DAMAGE' 'CRIMINAL TRESPASS' 'CRIM SEXUAL ASSAULT' 'PUBLIC PEACE VIOLATION' 'INTERFERENCE WITH PUBLIC OFFICER' 'PROSTITUTION' 'LIQUOR LAW VIOLATION' 'ARSON' 'OBSCENITY' 'CONCEALED CARRY LICENSE VIOLATION' 'KIDNAPPING' 'GAMBLING' 'INTIMIDATION' 'HUMAN TRAFFICKING' 'NON-CRIMINAL' 'OTHER NARCOTIC VIOLATION' 'PUBLIC INDECENCY'",
        "difficulty" : "easy",
        "type" : "data quality assessment"
    },
    {
        "question": "Print out the first five rows of the dataset in JSON format with the Date formatted in the european format.",
        "ground_truth": '{"ID":11662417,"Case Number":"JC232642","Date":"21/04/2019 12:30","Block":"009XX E 80TH ST","IUCR":"031A","Primary Type":"ROBBERY","Description":"ARMED - HANDGUN","Location Description":"RESIDENCE","Arrest":false,"Domestic":false,"Beat":631,"District":6,"Ward":8.0,"Community Area":44,"FBI Code":"03","X Coordinate":1184044.0,"Y Coordinate":1852159.0,"Year":2019,"Updated On":"14/09/2023 03:41","Latitude":41.749500329,"Longitude":-87.6011574,"Location":"(41.749500329, -87.6011574)"},'
                        '{"ID":12990873,"Case Number":"JG161829","Date":"17/08/2019 13:14","Block":"008XX N KARLOV AVE","IUCR":"1751","Primary Type":"OFFENSE INVOLVING CHILDREN","Description":"CRIMINAL SEXUAL ABUSE BY FAMILY MEMBER","Location Description":"RESIDENCE","Arrest":true,"Domestic":true,"Beat":1111,"District":11,"Ward":37.0,"Community Area":23,"FBI Code":"17","X Coordinate":1148899.0,"Y Coordinate":1905351.0,"Year":2019,"Updated On":"14/09/2023 03:41","Latitude":41.89621515,"Longitude":-87.728572048,"Location":"(41.89621515, -87.728572048)"},'
                        '{"ID":11630496,"Case Number":"JC193727","Date":"16/03/2019 11:35","Block":"045XX N LINCOLN AVE","IUCR":"0890","Primary Type":"THEFT","Description":"FROM BUILDING","Location Description":"BAR OR TAVERN","Arrest":false,"Domestic":false,"Beat":1911,"District":19,"Ward":47.0,"Community Area":4,"FBI Code":"06","X Coordinate":null,"Y Coordinate":null,"Year":2019,"Updated On":"23/03/2019 16:03","Latitude":null,"Longitude":null,"Location":null},'
                        '{"ID":11632505,"Case Number":"JC196841","Date":"20/03/2019 01:00","Block":"013XX W HOOD AVE","IUCR":"0810","Primary Type":"THEFT","Description":"OVER $500","Location Description":"OTHER","Arrest":false,"Domestic":false,"Beat":2433,"District":24,"Ward":48.0,"Community Area":77,"FBI Code":"06","X Coordinate":null,"Y Coordinate":null,"Year":2019,"Updated On":"27/03/2019 16:10","Latitude":null,"Longitude":null,"Location":null},'
                        '{"ID":11765926,"Case Number":"JC358692","Date":"21/07/2019 14:00","Block":"002XX W 87TH ST","IUCR":"0810","Primary Type":"THEFT","Description":"OVER $500","Location Description":"PARKING LOT / GARAGE (NON RESIDENTIAL)","Arrest":false,"Domestic":false,"Beat":622,"District":6,"Ward":21.0,"Community Area":44,"FBI Code":"06","X Coordinate":1176436.0,"Y Coordinate":1847222.0,"Year":2019,"Updated On":"15/09/2023 03:41","Latitude":41.736126864,"Longitude":-87.629184056,"Location":"(41.736126864, -87.629184056)"}',
        "difficulty" : "medium",
        "type" : "data cleaning"
    },
    {
        "question": "Print all IDs of the rows of the first ten rows where the crime was not Theft.",
        "ground_truth": "11662417, 12990873, 11885224, 13210699, 13211437, 13211898, 11842146",
        "difficulty" : "easy",
        "type" : "data cleaning"

    },
    {
        "question": "Create a new column IDXYEAR for each row by multiplying the ID with the YEAR column. Print out the first five values of IDXYEAR.",
        "ground_truth": "23546419923, 26228572587, 23481971424, 23486027595, 23755404594",
        "difficulty" : "easy",
        "type" : "data transformation and aggregation"
    },
    {
        "question": "Create a new column IDXYEAR for each row by multiplying the ID with the YEAR column. Print out the average of IDXYEAR over all rows.",
        "ground_truth": "The average of IDXYEAR is: 23688888976.91",
        "difficulty" : "medium",
        "type" : "data transformation and aggregation"
    },

    {
        "question": "Group the data by crime type and calculate the percentage of each group rounded to two values after the comma. Then calculate the difference of the group with the largest and the smallest percentage. Print out this delta percentage rounded to one value after the comma",
        "ground_truth": "23.9",
        "difficulty": "medium",
        "type": "data transformation and aggregation"
    },
    {
        "question": "Print the datatype of each column.",
        "ground_truth": "ID: int64, Case Number: object, Date: object, Block: object, IUCR: object, Primary Type: object, Description: object, Location Description: object, Arrest: bool, Domestic: bool, Beat: int64, District: int64, Ward: float64, Community Area: int64, FBI Code: object, X Coordinate: float64, Y Coordinate: float64, Year: int64, Updated On: object, Latitude: float64, Longitude: float64, Location: object",
        "difficulty" : "medium",
        "type" : "metadata extraction"
    }, 
    {
        "question": "Create a new column IDXYEAR for each row by multiplying the ID with the YEAR column. Does it make sense to create such a new column, answer with YES or NO",
        "ground_truth": "NO",
        "difficulty" : "hard",
        "type" : "data logic explanation"
    },

]
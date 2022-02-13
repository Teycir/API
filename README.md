# API:
http://127.0.0.1:5000/api/sk_ids/  ==> Extract list of 'SK_ID_CURR' from the DataFrame

http://127.0.0.1:5000/api/personal_data?SK_ID_CURR=384575  ==> Getting the data for the applicant

http://127.0.0.1:5000/api/features_desc  ==> Features description

http://127.0.0.1:5000/api/features_imp  ==> Features importance

http://127.0.0.1:5000/api/local_interpretation?SK_ID_CURR=384575   ==> Computation of the prediction, bias and contribs from surrogate model

Here is my classification report and confusion matrix for after base training. No fine tuning yet. 

                 precision  recall   f1-score    support

       akiec       0.25      0.57      0.35        44
         bcc       0.43      0.52      0.47        69
         bkl       0.57      0.55      0.56       146
          df       0.20      0.73      0.31        15
         mel       0.39      0.60      0.47       148
          nv       0.96      0.74      0.84       894
        vasc       0.49      0.89      0.63        19

    accuracy                           0.69      1335
    macro avg      0.47      0.66      0.52      1335
    weighted avg   0.79      0.69      0.72      1335

[[ 25   7   5   4   3   0   0]
 [ 12  36   5   5   3   3   5]
 [ 18  13  81   5  21   7   1]
 [  3   0   1  11   0   0   0]
 [ 15   6  12   6  89  20   0]
 [ 27  21  35  25 111 663  12]
 [  0   0   2   0   0   0  17]]

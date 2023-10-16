# Price-Prediction-Model
Model to predict the closing price of a stock on the next trading day and dashboard plotting the accuracy of the predictions over time.

This entire project is directed at building the following:

1. A baseline automated predictive model that creates predictions for one variable based on the historical outcome of a few other variables. This model isn't meant to be incredibly accurate for the first go around. Predictive quality is something I plan to work on more in-depth later. The data I am training the model on currently only has three to four independent variables potentially meaningful to prediction.

2. A standard dashboard that presents a user with the prediction for a desired variable for the next time increment (the price of a stock on the next trading day)

3. A user interface where a user can upload simple clean data, choose the dependent variable, create a dashboard with predictions incremented by some measure of time, and a visualization tracking the historical accuracy of the prediction.

The first few weeks of effort will be spent on building, refining, and validating the first two points. Once complete, I plan to move to point three.


Model Automation Instructions:

I used cron to create the automation of the next day closing price. In terminal, run:
 
  crontab -e

When cron opens, input and save (requires saving the Predictions file as a .py. This is included in the repo):

  nate@Nates-MacBook-Air ~ % chmod +x /Users/nate/Desktop/Prediction\ Dashboard/Price\ Prediction.py

  

  

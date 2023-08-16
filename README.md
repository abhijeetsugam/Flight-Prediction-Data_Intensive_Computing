# Flight-Prediction-Data_Intensive_Computing 
# Phase_1
EDA Done ✅✅


# Motivation: 
The motivation behind the Flight Price Prediction problem statement is to provide 
customers with accurate and reliable information about the cost of air travel. Flight prices 
can be affected by a variety of factors, including the time of year, day of the week, the 
destination, the airline, and the availability of seats. These factors can make it difficult for 
travellers to determine the best time to book a flight or the best airline to use for their trip. 
In later part of our project, using machine learning algorithms we intend to predict flight 
prices, to provide customers with valuable information that can help them save money and 
make more informed decisions about their travel plans. This can benefit not only customers 
but also airlines and travel agencies, who can use the predictions to optimize their pricing 
strategies and improve customer satisfaction. To cater the needs of customer we have 
developed a website as well, which allows user to predict the possible flight fare based on
factors which impacts it directly. We have also showcased insights, analysing our dataset to 
show which factor impacts price in what ways. End user will get fair idea of when to perform 
their booking to get best price.


# Goal: 
Overall, the goal behind the Flight Price Prediction problem statement is to provide 
travellers with better transparency and pricing accuracy while enabling businesses to 
optimize their pricing strategies and improve their competitiveness in the market


# Dataset Description: 

The Flight Price Prediction dataset available on Kaggle, is a comprehensive collection of 
flight data that can be used for training and testing machine learning models to predict 
flight prices.

The dataset includes around 30,000 records for various domestic airlines in India. The data 
was collected from different sources, including airline websites and online travel agencies. 
The various features of the cleaned dataset are explained below: 
Airline: The name of the airline company is stored in the airline column. It is a categorical 
feature having 6 different airlines.

## Flight: 
Flight stores information regarding the plane's flight code. It is a categorical feature.
Source City: City from which the flight takes off. It is a categorical feature having 6 unique 
cities.

## Departure Time: 
This is a derived categorical feature obtained created by grouping periods into bins. It stores information about the departure time and have 6 unique time 
labels.

## Stops: 
A categorical feature with 3 distinct values that stores the number of stops between 
the source and destination cities.

## Arrival Time: 
This is a derived categorical feature created by grouping time intervals into 
bins. It has six distinct time labels and keeps information about the arrival time.
Destination City: City where the flight will land. It is a categorical feature having 6 unique 
cities.

## Class: 
A categorical feature that contains information on seat class; it has two distinct 
values: Business and Economy.

## Duration: 
A continuous feature that displays the overall amount of time it takes to travel 
between cities in hours.

## Days Left: 
This is a derived characteristic that is calculated by subtracting the trip date by 
the booking date.

## Price: 
Target variable stores information of the ticket price.

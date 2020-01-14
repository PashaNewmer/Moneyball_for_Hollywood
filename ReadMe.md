# Moneyball for Hollywood  
  
Capstone for General Assembly Data Science Immersive by Pasha Newmer

## Problem Statement   
   
How to recognize a financially successful movie at the very starting point of its production process by analyzing existing data, using NLP models on keywords from the plot, taglines, cast and crew, as well as entire movie scripts.   
In this project, I focused on finding the answer to the question that every moviemaker-producer-investor asks themselves: "Should I start working on this movie or choose another one? Will this project be profitable to me and my partners (company)?"  


## Datasets

First [Data set](../Data/movies_metadata.csv) of about 45,000 movies with metadata was collected by [Rounak Banik](https://www.kaggle.com/rounakbanik) from TMDB.  The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.  
This dataset consists of the following files:  

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.  

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.  

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.  
  
I extracted needed information out of it in an EDA notebook and saved it is as [Clean_Dataset](./Data/Clean_movies.csv).   
  
Second [Data set](./Data/Script_budget.csv) of the [movie scripts](https://www.imsdb.com/all%20scripts/) from www.imsdb.com in form of 1093 .txt files, it's [budgets and revenues](./Data/Budgets_5686 - Budgets.csv) from www.the-numbers.com/movie/budgets in combination with budgets and revenues from [metadata set](./Data/movies_metadata.csv). 
  
After merging on the 'title' column I got the dataset of scripts, titles, budgets and revenues for 642 movies and saved it as [Script_budget](./Data/Script_budget.csv) dataset.
  
  
## Data Dictionary

#### Data dictionary for Clean_movies dataset

|Feature|Type|Description|
|---|---|---| 
|**belongs_to_collection**|*int*|Binary column that indicates if the movie belongs to collection|
|**budget**|*float*|The budget of the movie in dollars|
|**genres**|*object*|A list of all the genres associated with the movie|
|**homepage**|*int*|Binary column that indicates if the movie has a homepage|
|**overview**|*object*|A brief overview of the movie|
|**popularity**|*float*|The Popularity Score assigned by TMDB|
|**production_companies**|*object*|A list of production companies involved with the making of the movie|
|**production_countries**|*object*|A list of countries where the movie was shot|
|**release_date**|*datetime*|Release Date of the movie|
|**revenue**|*float*|The total revenue of the movie in US dollars|
|**runtime**|*float*|Duration of the movie in minutes|
|**spoken_languages**|*object*|A list of spoken languages in the movie|  
|**tagline**|*object*|The tagline of the movie|  
|**title**|*object*|The Official Title of the movie|  
|**vote_average**|*float*|The average voting rating of the movie, as counted by TMDB|  
|**vote_count**|*float*|The number of votes, as counted by TMDB|  
|**keywords**|*object*|The movie plot keywords|  
|**cast**|*object*|Names of the cast of the movie|  
|**crew**|*object*|Names of the crew of the movie|
  
#### Data dictionary for Script_budget dataset

|Feature|Type|Description|
|---|---|---| 
|**script**|*object*|Entire movie script|
|**title**|*object*|The Official Title of the movie|
|**budget**|*float*|The budget of the movie in US dollars|
|**revenue**|*float*|The total revenue of the movie in US dollars|
  
  
# Executive Summary

None of the most advanced models I used were able to identify any patterns for predicting financial success with such a thin matter as the art of writing the movie script. There is still a big room for experiments, with different combinations of features, stop-words, and parameters. Mining more data, having a team of people and much more computational power, having some linguists and writers on the team definitely will help. 
  
There might be separate models for each genre, to find the pattern in the scripts of the same type. 
  
Another possible experiment - to create a dataset with movie scripts, divided not by the genre as we used to know them, but by the story types, and there is 10 of them according to [Blake Snyder](http://www.savethecat.com/). Divided by what exactly going on with the main character and how he changes during the movie.
Short example: Die Hard, Schindler's List and Terminator is the same type of story, even it is completely different genres. Blake Snyder calls this type - "Dude with a problem". It consists of an INNOCENT HERO, SUDDEN EVENT and LIFE OR DEATH BATTLE. Absolutely ordinary person (policeman, director of the factory in Poland or waiter in the diner) got into absolutely extraordinary situation - terrorists capture the building, nazis dragging Jewish friends to extermination camp or robot from the future (with accent!) trying to kill her and her unborn (and not even conceived yet) child!
  
So the scripts might be divided by the story types and then processed through the models. Than extracting just a verbs will helps to determine what's going on with the main character. And if this character's type belongs to this story's type, then we will be able to say that this movie will be successful. And if you will remember a lot of successful movies with most successful actors - their types match the types of stories they play. 
  
Also, patterns might be found while exploring the structures of the plots. Most successful movie's plots matching with "the perfect structure" with 3 main parts and other important bits. Back in a day I did some analysis about that matter and found that most of the successful movies do that. And the champion of this "matching" is Steven Spielberg, one of the most successful directors ever. 
  
Again this is just a theory, that requires a lot of men/hours and computational power to be proved or disproved. 
  
I will gladly continue my research in this direction, because the fruit at the end is priceless, especially for movie and data enthusiasts like myself in particular and moviemakers in general. 
As for now we still can learn a lot from the old good exploratory analysis and visualizations.
  
In the first notebook with EDA, I found that all the movies might be divided by 3 huge clusters. 
  
First one contains super-expensive blockbusters, animations, adventures, fantasy, sci-fi and family movies in general. Mostly been shot by the same companies, same directors with the same actors and released right before or at the beginning of the summer, so all kids and their young parents can enjoy it. They generating the biggest revenues of all times that might be compared with the GDP of come countries. And they definitely worth it. But usually this revenues equal to just a few of its budgets, which is still a tremendous amount of money, but the risks are high too. 
  
The second cluster contains mid-budget movies - dramas, comedies, actions, and historical films, made by directors that already proved themselves, with a good and expensive cast, released near the autumn or December, probably closer to awards ceremonies. Because a lot of them definitely represents a high artistic value. Most of the legendary movies we know, that survived decades and still fascinates hundreds of millions of people is belong to that cluster. They accumulate very high revenues and have the perfect balance between budgets, revenues and amounts of the budgets that returned to their creators. But they still risky for investors, because art is very unpredictable and subjective. 
  
Finally, the third cluster contains low-budget horrors (mostly), mysteries and even dramas, that were written by unknown writers, shot by unknown directors with unknown actors (and sometimes it is the same person) with unknown devices (even smartphones this day). But because their budgets are so low, their success sometimes makes very jealous even those, who sold their bitcoins in December of 2017. The best examples are "Paranormal Activity" and "The Blair Witch Project", with 15k or 60k budgets and 200+ millions of dollars revenue. With no special effects and interesting twisted plots, these movies serve as a great springboard for everyone, who took part in its creation. 
  
Usually, creators (writers, directors, producers and actors) of successful low budget movies demonstrate perfect transition through this clusters, from third to the first one. My favorite examples of such transitions are Sylvester Stallone, Matt Damon, Harrison Ford, Arnold Schwarzenegger, etc. I would like to bring a visual example of it, based on the numbers: 
  
  
  
#### James Cameron 
  
  
![image](./Visualizations/James_cameron.png) 
  
  
#### Darren Aronofsky 
  
  
![image](./Visualizations/Aranofsky.png) 
  
With this being said I wish each and everyone who involved into the process of the creation of the movies - inspiration and courage on their not easy path of bringing joy and happiness, excitement and inspiration to the people all over the world.
  
  
P.S. And don't forget to make sure that your story will have a continuation in order to be part of the collection... 
  
P.P.S. And of course make a homepage for your movie!  
  


                                                                                    to be continued...

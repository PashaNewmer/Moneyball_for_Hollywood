# Moneyball for Hollywood  
  
Capstone for General Assembly Data Science Immersive by Pasha Newmer

## Problem Statement   
   
How to recognize financially successful movie at the very starting point of it's production process by analyzing existing data, using NLP models on keywords from the plots, taglines, cast and crew, as well as entire movie scripts.   
In this project I worked on finding the answer for the question, that every moviemaker-producer-investor asks himself: "Should I start working on this movie or choose another one? Will this one brings some profit to me and my partners(company)?"  


## Datasets

First [Data set](./data/movies_metadata.csv) of about 45000 movies with metadata was collected by [Rounak Banik](https://www.kaggle.com/rounakbanik) from TMDB. These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.  
This dataset consists of the following files:  

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.  

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.  

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.  
  
I extracted the information out of it in EDA notebook based on our needs and saved is as [Clean_Dataset](./Data/Clean_movies.csv).  
  
Second [Data set](./Data/Script_budget.csv) of the [movie scripts](https://www.imsdb.com/all%20scripts/) from www.imsdb.com in form of 1093 .txt files, it's [budgets and revenues](./Data/Budgets_5686 - Budgets.csv) from www.the-numbers.com/movie/budgets in combination with budgets and revenues from [metadata set](./Data/movies_metadata.csv)  
  
After merging on the 'title' column I got dataset of scripts, titles, budgets and revenues for 642 movies and saved it as [Script_budget](./Data/Script_budget.csv) dataset.
  
  
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
|**budget**|*float*|The budget of the movie in dollars|
|**revenue**|*float*|The total revenue of the movie in US dollars|
  
# Executive Summary

None of the most advanced models I used were able to identify any patterns for predicting financial success with such a thin metter as the art of writing the movie script. As I said before there is still a big room for experiments, with different combinations of features, stop-words and parameters. Mining more data, having a team of people and much more computational power, having some linguists and writers on the team definitely might helps.  
  
There might be separate models for each genre, to find the pattern in the scripts of the same type.  
  
Or even create dataset with movie scripts, divided not by the genre as we used to know them, but by the story types, and there is 10 of them according to [Blake Snyder](http://www.savethecat.com/). Divided by what exactly going on with the main character and how he change during the movie. Short example: Die Hard, Schindler's List and Terminator is the same type of the story, even it is completely different genres. Blake Snyder calls this type - "Dude with a problem". It consist INNOCENT HERO, SUDDEN EVENT and LIFE OR DEATH BATTLE Absolutely ordinary person (police man, director of some factory in Poland or weiter in the diner) got into absolutely extraordinary situation - terrorists capture the building, nazis dragging jewish friends to extermination camp or robot from the future (with accent!) trying to kill her and her unborn (and not even conceived yet) child!  
  
So the scripts might be divided by the story types and then processed through the models. Than extracting just a verbs will helps to determine what's going on with the main character. And if this character's type belongs to this story's type, then we will be able to say that this movie will be successful. And if you will remember a lot of successful movies with most successful actors - their types totally match the types of the stories they play.  
  
Also patterns might be found while exploring the structures of the plots. Most successful movie's plots matching with "the perfect structure" with 3 main parts and other important bits. Back in a days I did some analysis about that matter and found that most of the successful movies does that. And the champion of this "matching" is... Steven Spielberg, one of the most successful directors ever.  
  
Again this is just a theories, that requires a lot of men/hours and computational power to be proved or disproved.  
  
I will gladly continue my research in this direction, because the fruit at the end is priceless, especially for movie and data enthusiasts like myself in particular and moviemakers in general.  
As for now we still can learn a lot from the old good exploratory analysis and visualizations combined with life experience, knowledge of the industry and common sense.  
  
In the first notebook with EDA I found that all the movies we explored might be divided by 3 huge clusters.  
  
First one contains super-expensive blockbusters, animations, anventures, fantasy, sci-fi and family movies in general. Mostly been shot by the same companies, same directors with the same actors and released right before or at the beginning of the summer, so all kids and their young parents can enjoy it. They generating the biggest revenues of all times that might be compared with GDP of come countries. And they definitely worth it. But usually this revenues equal to just few of it's budgets, which is still tremendous amount of money, but the risks is high too.  
  
Second cluster contains mid-budget movies - dramas, comedies, actions and historical films, made by directors that already proved themselves, with good and expensive cast, released near the autumn or December, probably closer to awards ceremonies. Because lot of them definitely represents a high artistic value. Most of the legendary movies we know, that survived decades and still fascinates hundreds of millions people is belong to that claster. They accumulate very high revenues and I would say that they have the perfect balance between budgets, revenues and amounts of the budgets that returns to their creators. But they still risky for investors, because art is very unpredictable and subjective.  
  
Finally the third cluster contains low-budget horrors (mostly), mysteries and even dramas, that were written by unknown writers, shot by unknown directors with unknown actors (and sometimes it is the same person) with unknown devices (even smartphones this days). But! Because their budgets is so low, their success sometimes makes very jealous even those, who sold their bitcoins in December of 2017. The best examples is "Paranormal activity" and "The Blair Witch Project", with 15k or 60k budgets and 200+ millions of dollars revenue! With no special effects and interesting twisted plots this movies serves as a great springboard for everyone, who taking part in it's creation.  
  
Usually moviemakers (writers, directors, producers and actors) of successful low budget movies demonstrate perfect transition through this clasters, from third to the first one. My favorite examples of such transitions is: Sylvester Stallone, Matt Damon, Harrison Ford, Arnold Schwarzenegger of course (and not only because of accent and immigration history) and many many others. I would like to bring visual example of it, based on the numbers:  
  
  
  
#### James Cameron  
  
  
![image](./Visualizations/James_cameron.png)  
  
  
#### Darren Aronofsky  
  
  
![image](./Visualizations/Aranofsky.png)  
  
With this being said I wish each and everyone who involved into the process of the creation of the movies - inspiration and courage on their not easy path of bringing joy and happiness, excitement and inspiration to the people all over the world. And I, personally, will definitely keep doing this.  
  
  
P.S. And don't forget to make sure that your story will have a continuation in order to be the part of collection...  
  
P.P.S. And of course make a homepage for your movie!

                                                                                    to be continued...

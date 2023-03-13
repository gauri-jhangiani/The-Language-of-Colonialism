# catch-all working file for your project.


import math
import pandas as pd
from datetime import datetime
import statistics
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
import os
import string
import matplotlib.pyplot as plt


class Corpus(object):
    def __init__(self):
        # =========
        # IMPORTANT FLAGS SET HERE
        self.query = 'bombay'
        # self.category is either False or a metadata category to sort by. two examples:
        #self.category = False
        self.category = 'speaker'
        self.corpus_folder = 'speeches/'
        # self.graph_type can be 'line' or 'scatter'
        self.graph_type = 'bar'
        # to divide the corpus, give it a tuple with a category and value. Otherwise, uncomment the following line to use the whole corpus
        self.corpus_subdivision = False
        #self.corpus_subdivision = ('location','Mumbai')
        # =========
        self.filenames = self.all_files()
        self.texts = [Text(filename, self.query, self.category) for filename in self.filenames]
        self.texts.sort(key=lambda x: x.date)
        if self.corpus_subdivision:
            self.texts = self.get_subset_by_metadata(self.corpus_subdivision[0], self.corpus_subdivision[1])
        # self.texts.sort(key=lambda x: x.token_length)
        #the part commented out above if used in place of sorting by date will give an ordered list of all speeches by length
        self.speech_titles=[text.speech_name for text in self.texts]
        self.all_locations= [text.location for text in self.texts]
        self.location_wf= nltk.FreqDist(self.all_locations)
        self.location_wf_list=self.location_wf.most_common()
        self.all_audiences= [text.audience for text in self.texts]
        self.audiences_wf= nltk.FreqDist(self.all_audiences)
        self.audiences_wf_list=self.audiences_wf.most_common(50)
        self.bigrams_across_corpus= self.pull_out_bigrams()
        self.collocations_across_corpus=self.pull_out_collocations()
        self.collocations_freqdist=nltk.FreqDist(self.collocations_across_corpus)
        self.bigrams_freqdist = nltk.FreqDist(self.bigrams_across_corpus)
        self.bigrams_top20=self.bigrams_freqdist.most_common(30)
        self.word_count=[text.token_length for text in self.texts]
        self.speakers=[text.speaker for text in self.texts]
        self.all_dates=[text.text_date for text in self.texts]
        self.all_sentence_lengths=[text.sentence_lengths for text in self.texts]
        self.all_average_sentence_lengths=[text.average_sentence_length for text in self.texts]
        self.all_lexical_diversity=[text.lexical_diversity for text in self.texts]
        self.words_across_corpus=self.pull_out_all_words()
        self.all_words_count=len(self.words_across_corpus)
        self.words_freqdist=nltk.FreqDist(self.words_across_corpus)
        
        ### collect processes for graphing data
        # produce a dataframe with all of our word frequencies and the word counts of the query we want
        self.data_to_graph = self.get_data_over_time(self.texts, self.query)
        print(self.data_to_graph)
        self.overall_top_words = nltk.FreqDist(self.pull_out_all_words()).most_common(25)
        if self.category:
            # # one for each category in the metadata sheet
            self.category_frames = self.divide_into_categories(self.data_to_graph)
            # check into how it's doing these by month
            self.regularized_category_frames = [self.regularize_data_frame(df, self.query) for df in self.category_frames]
            self.regularized_category_frames.sort(key=self.sort_elem)
            self.graph(self.regularized_category_frames)
        else:
            self.regularized_frame = self.regularize_data_frame(self.data_to_graph, self.query)
            self.graph(self.regularized_frame)

    def get_subset_by_metadata(self, key, value):
        """if you want to pull out a sub_corpus, you would run this sub_corpus = this_corpus.get_subset_by_metadata('blog','ghost')"""
        return [text for text in self.texts if getattr(text, key) == value]

    def most_common_bigrams_in_files(self,query,position):
        """Pass it a word and what position in the bigram it's in. Note for the future because it's unclear - position can be 0 or 1"""
        list_of_bigrams = [term for term in self.bigrams_freqdist if term[position] == query]
        for term in list_of_bigrams:
            if self.bigrams_freqdist[term] > 1:
                print('------')
                list_of_files_with_hits = [text.filename for text in self.texts if term in text.bigrams]
                print(term)
                print(self.bigrams_freqdist[term])
                for filename in list_of_files_with_hits:
                    print(filename)


    def pull_out_all_words(self):
        all_words = []
        for text in self.texts:
            all_words.extend(text.stop_and_punctuation_removed)
        return all_words
        
# To graph an NLTK plot:
        # fig = plt.figure(figsize = (10,4))
        # corpus.audiences_wf.plot(10)
        # plt.show()
        # fig.savefig('freqDist.png', bbox_inches = "tight")

    def graph(self, data):
        """given a set of dataframes to graph, graph them"""
        plt.clf()
        #ISSUES: not sharing x axes, graphing zero values? empty plots
        plt.style.use('seaborn-whitegrid')
        plt.clf()
        # should be a fixed number of metadata options/colors
        # more than six is going to be wonky
        colors = {0:'b', 1: 'g', 2: 'r', 3: 'm', 4: 'y', 5: 'c', 6: 'k'}
        color_count = 0
        fig_counter = 0
        if type(data) == list:
            # if we have a category, we're going to be handed a stack of tables to graph individually.
            # This is close. Need to keep common X and Y axes and maybe figure out how to drop empty dataframes from the sample. Seems like it might be struggling to graph things that only have one hit (which is most!)
            ### throw away those speeches that only have zero
            data = [df for df in data if not (df['DATA'] == 0).all()]
            # throw away those categories that only have one speech if we are doing a line graph
            if self.graph_type in ['line']:
                data = [df for df in data if (df.shape[0] > 1)]
            if self.graph_type == 'bar':
                combined_data = pd.concat(data)
                agg_functions = {'DATA': 'sum', 'DATE': 'min'}
                combined_data = combined_data.groupby(combined_data['CATEGORY']).aggregate(agg_functions).reset_index()
                # can rename x and y here if you want by changing the value in each key value pair (the second one)
                column_mappings = {'CATEGORY': self.category, 'DATA': 'counts'}
                combined_data = combined_data.rename(columns=column_mappings)
                combined_data = combined_data.sort_values(by="DATE")
                print(combined_data)
                combined_data.plot.bar(x=self.category, y='counts', legend=None)
                plt.gca()
                if self.category:
                    plt.title('Counts for term"' + self.query + '" by ' + self.category)
                else:
                    plt.title('Counts for term "' + self.query + '" by date')
                plt.savefig('results/bar_graph.png',bbox_inches="tight")
            else:
                # set the y value to 10 plus the max value
                max_y_val = max([df['DATA'].max() for df in data]) + 10
                # set the x values to be the min and max dates
                min_x_val = min([df['DATE'].min() for df in data])
                max_x_val = max([df['DATE'].max() for df in data])
                # loop through every category and graph the remaining
                for dataframe in data:
                    # if color count is divisible by 7 we are at the max number of colors for each graph, so graph what we have and start over with a clean graph
                    if color_count % 7 == 0 and color_count != 0:
                        # set the legend location
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True)
                        # set the x tickmarks to be 90 degrees
                        plt.xticks(rotation=90)
                        if self.category:
                            # plot title is set here
                            plt.title('Counts for term"' + self.query + '" by ' + self.category)
                            # save our current graph using the fig counter variable
                            plt.savefig("results/results_graph_" + str(fig_counter) + ".png",bbox_inches="tight")
                        # add one to the fig_counter variable so we can keep making new graphs 
                        fig_counter += 1
                        # reset the color variable so that the new graph starts over
                        color_count = 0
                        # now that we've saved this graph, clear the plot so we can start over
                        plt.clf()
                    # get the axes
                    ax = plt.gca()
                    # set max x and y values
                    ax.set_ylim([0, max_y_val])
                    ax.set_xlim([min_x_val, max_x_val])
                    # plot as scatter or line graph depending on the flag
                    if self.graph_type == 'line':
                        plt.plot(dataframe['DATE'],dataframe['DATA'].values, color=colors[color_count],label=dataframe.CATEGORY.iloc[0])
                    elif self.graph_type == 'scatter':
                        plt.scatter(dataframe['DATE'],dataframe['DATA'].values, color=colors[color_count],label=dataframe.CATEGORY.iloc[0])
                    # add one to the color counter        
                    color_count +=1
                    print(dataframe)
                    # plt.show()
                # get the current graph
                ax = plt.gca()
                # set the limits for the y and x axes
                ax.set_ylim([0, max_y_val])
                ax.set_xlim([min_x_val, max_x_val])
                # set the legend
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=True, shadow=True)
                # set the layout to be tight
                plt.tight_layout()
                # set the x tick marks to be rotated 90 degrees
                plt.xticks(rotation=90)
                # plot title is set here
                if self.category:
                    plt.title('Counts for term"' + self.query + '" by ' + self.category)
                else:
                    plt.title('Counts for term "' + self.query + '" by date')
                plt.savefig("results/results_graph_" + str(fig_counter) + ".png",bbox_inches="tight")
        else:
            # if no category, we're just using one search query and all the data
            if self.graph_type == 'line':
                plt.plot(data['DATE'],data['DATA'].values, color=colors[color_count])
            elif self.graph_type == 'scatter':
                plt.scatter(data['DATE'],data['DATA'].values, color=colors[color_count])
            elif self.graph_type == 'bar':
                plt.bar(data['CATEGORY'],data['DATA'].values, color=colors[color_count])
            color_count += 1
            plt.title('Counts for term "' + self.query + '" by date')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True, shadow=True)
            plt.xticks(rotation=90)
            plt.savefig("results/results_graph.png", bbox_inches="tight")
            plt.clf()


    def divide_into_categories(self, df):
        """takes a single dataframe, tagged for categories, and divides into separate dataframes based on each type"""
        return [df[df.CATEGORY == this_category] for this_category in set(df.CATEGORY.values)]

    def get_data_over_time(self, texts, query):
        """Handles the compilation and merging of data over time"""
        df = pd.DataFrame()
        # df['AUDIENCE'] = [getattr(text, text.audience) for text in texts]
        if self.category:
            df['CATEGORY'] = [getattr(text, text.category) for text in texts]
        df['AUDIENCE'] = [text.audience for text in texts]
        df['LOCATION'] = [text.location for text in texts]
        df['SPEAKER'] = [text.speaker for text in texts]
        df['LOCATION'] = [text.location for text in texts]
        df['DATE'] = [text.date for text in texts]
        if query == 'LD':
            df['DATA'] = [text.lexical_diversity for text in texts]
        elif ' ' in query:
            bigram_query = list(nltk.bigrams(nltk.word_tokenize(query)))[0]
            df['DATA'] = [text.bigrams_freqdist[bigram_query] for text in texts]
        elif query:
            df['DATA'] = [text.word_frequencies[query] for text in texts]
        else:
            raise NameError('No query given')
        return df

    def regularize_data_frame(self, df, query):
        """only graphs unique shared days atm. this is where we would average per month if we wanted"""
        unique_dates = set(df.DATE.values)
        converted_df = pd.DataFrame()
        converted_df['DATE'] = [date for date in unique_dates]
        if self.category:
            converted_df['CATEGORY'] = df.CATEGORY.iloc[0]
        if query == 'LD':
            # only average LD for now, otherwise add them
            converted_df['DATA'] = [df[df['DATE'] == date]['DATA'].mean() for date in unique_dates]
        else:
            converted_df['DATA'] = [df[df['DATE'] == date]['DATA'].sum() for date in unique_dates]
        return converted_df.sort_values(by=['DATE'])
    
    def sort_elem(self,elem):
        return elem['DATE'].min()

    def pull_out_bigrams(self):
        corpus_bigrams = []
        for text in self.texts:
            corpus_bigrams.extend(text.bigrams)
        return corpus_bigrams
    def pull_out_collocations(self):
        corpus_collocations = []
        for text in self.texts:
            corpus_collocations.extend(text.collocations)
        return corpus_collocations

    
    def find_keyword_in_corpus(self,query):
        results = [(text.filename, text.find_keyword(query)) for text in self.texts if len(text.find_keyword(query)) > 0]
        for result in results:
            print('=======')
            print(result[0])
            print(result[1])
        return results
        
        
    

    def sort_by_date(self):
        return self.texts.sort(key=lambda x: x.date)

    def sort_by_token_length(self):
        return self.texts.sort(key=lambda x: x.token_length)

    def word_over_time(self, word):
        return [(text.speaker, text.word_frequencies[word]) for text in self.texts]
        
    
    

    

    def all_files(self):
            """given a directory, return the filenames in it"""
            texts = []
            for (root, _, files) in os.walk(self.corpus_folder):
                for fn in files:
                    if fn[0] == '.': # ignore dot files
                        pass
                    else:
                        path = os.path.join(root, fn)
                        texts.append(path)
            return texts
    
def write_to_file(stuff_to_print, the_file_to_print_to):
    """note: analysis.write_to_file(corpus.find_keyword_in_corpus("years"), "years_in_context_search.txt")"""
    with open('results/' + the_file_to_print_to, 'w') as f_out:
        if type(stuff_to_print) == list:
            for result in stuff_to_print:
                f_out.write('=======\n')
                f_out.write(result[0])
                f_out.write('\n')
                f_out.write(str(result[1]))
                f_out.write('\n')
        else:
            f_out.write(str(stuff_to_print))

#to print into a file at the corpus level:
#lady= Corpus.word_over_time(Corpus, "lady")
#with open("lady.txt", "w") as f:
    #print(lady, file=f)

     #with open("lady.txt", "w") as f:
            #print(result, file=f)

#lady= Corpus.find_keyword_in_corpus(Corpus, "lady")
#with open("lady.txt", "w") as f:
   # print(lady, file=f)


class Text(object):
    def __init__(self, filename, query, category):
        # adjectives here
        self.filename = filename
        self.category = category
        print('=======')
        print(self.filename)
        self.raw_text_lines = self.read_file()
        # note this is where we set the starting line
        self.raw_text = ' '.join(self.raw_text_lines[10:])
        self.raw_tokens = nltk.word_tokenize(self.raw_text)
        self.clean_tokens = nltk.word_tokenize(self.raw_text.lower())
        self.vocabulary = set(self.clean_tokens)
        self.stopwords_removed = [w for w in self.clean_tokens if w not in nltk.corpus.stopwords.words('english')]
        self.stop_and_punctuation_removed = [w for w in self.stopwords_removed if w not in string.punctuation]
        self.lexical_diversity = len(self.vocabulary)/len(self.clean_tokens)
        self.sentences_in_characters = nltk.sent_tokenize(self.raw_text)
        self.sentences_in_tokens = [nltk.word_tokenize(sentence) for sentence in self.sentences_in_characters]
        self.sentence_lengths = [len(sentence) for sentence in self.sentences_in_tokens]
        self.average_sentence_length = statistics.mean(self.sentence_lengths)
        # self.whatever_you_want = the_code_that_generates_that_result
        self.word_frequencies = nltk.FreqDist(self.stop_and_punctuation_removed)
        self.nltk_version_text = nltk.Text(self.stopwords_removed)
        self.bigrams = list(nltk.bigrams(self.stop_and_punctuation_removed))
        self.bigrams_freqdist = nltk.FreqDist(self.bigrams)
        self.collocations = self.nltk_version_text.collocation_list()
        self.speech_name=self.raw_text_lines[1].replace('\n','').split(': ')[1]
        self.speaker= self.raw_text_lines[2].replace('\n','').split(': ')[1]
        self.audience=self.raw_text_lines[3].replace('\n','').split(': ')[1]
        self.text_date=self.raw_text_lines[4].replace('\n','').split(': ')[1]
        print(self.text_date)
        self.date = datetime.strptime(self.text_date, "%d %B %Y")
        self.location=self.raw_text_lines[5].replace('\n','').split(': ')[1]
        self.token_length= len(self.clean_tokens)
        self.character_length=len(self.raw_text)
       
    
    def find_keyword(self, keyword):
        return [w.replace('\n','') for w in self.sentences_in_characters if keyword in nltk.word_tokenize(w)]
    

   
    
    # def function_template(self):
    #     the_value = code to generate the value
    #     return the_value

    def read_file(self):
        with open(self.filename, 'r') as file_in:
            raw_text = file_in.readlines()
        return raw_text

#to print results from the Class
#coll= corpus.texts[3].collocations
#with open("lady.txt", "w") as f:
    #print(coll, file=f)

def main():
    # if run from terminal, it will run whatever is here
    corpus = Corpus()

# this allows you to import the classes as a module. it uses the special built-in variable __name__ set to the value "__main__" if the module is being run as the main program

if __name__ == "__main__":
    main()

# to use with the python interpreter
# $ python3
# >>> import analysis
# >>> corpus = analysis.Corpus()
# >>> corpus.texts
# >>> corpus.texts[0].raw_text

# if you have changed something in this file while working in the interpreter
# importlib is a library that lets you reload modules
# >>> import importlib
# use importlib to reload our analysis file
# >>> importlib.reload(analysis)
# remake our corpus
# >>> corpus = analysis.Corpus()

# Keyword in context
# after you import your corpus you can search for a keyword by giving it one - 
# corpus.find_keyword_in_corpus('Bombay')

#with open("file.txt", "a") as f:
#   print(self.bigrams_across_corpus, file=f)

# You no longer have to pass "speeches/" every time when you build the corpus.
# Instead we rely pretty heavily on the section I made at the top called IMPORTANT FLAGS, where you set the query you're interested in, the corpus folder, and whether or not to group things by metadata category.

# How to troubleshoot. First, take a look at the guidance at https://humanitiesprogramming.github.io/
# Then, try to determine where your error is. Is it in the corpus data itself? The code? or do you not understand how a particular function works?
# It's often helpful to add print(self.VARIABLENAME) to various parts of the class.
# i.e. - print(self.filename) in the text class will print out all the files as you go, so you will see the exact filename that is erroring. from there you can add different attribute names to start to troubleshoot where the error is coming from.

#graph word count over speaker
#import matplotlib.pyplot as plt
#plt.bar(corpus.speakers,corpus.word_count)
#plt.title('Word count by Speaker')
#plt.xticks(rotation=90)
#plt.xlabel('speakers')
#plt.ylabel('word count')
#plt.show()

#graph word count over time
#import matplotlib.pyplot as plt
#plt.bar(corpus.all_dates, corpus.word_count)
#plt.xlabel('dates')
#plt.ylabel('word count')
#plt.show()

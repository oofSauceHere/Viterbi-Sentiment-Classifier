"""
Note: The viterbi implementation for sentiment analysis is from user esmondchuah on Github. My addition
to the algorithm is a way to determine the sentiment of whole sentences by taking into account the frequency
of words
"""

class Word:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
    def add(self):
        self.freq += 1

import sys
sys.setrecursionlimit(2000)

possible_states = ["O", "B-positive", "I-positive", "B-neutral", "I-neutral", "B-negative", "I-negative"]


# a modified Data_processor class which performs some pre-processing on the training data before returning the formatted array of data.
class Data_processor:
    def __init__(self, path):
        self.datal = []
        self.data = []
        self.file = open(path, 'r', encoding="utf8")
        for i in self.file.read().split("\n\n") :
            sentence = []
            unlabeled = []
            for j in i.split("\n"):
                if j != "":
                    unlabeled.append(j)
                    word = j.split(" ")[0].lower()
                    if len(word) > 5 and word[:4] == "http":
                        word = word[:4]
                    if len(j.split(" ")) >1:
                        sentence.append(word + " " + j.split(" ")[-1])
                    else:
                        sentence.append(word)
            if len(sentence) > 0:
                self.data.append(sentence)
                self.datal.append(unlabeled)
        self.file.close()

def word_cloud(dataset):
    keywords = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            wd = dataset[i][j]
            wd = wd.lower()
            isIn = False
            for w in keywords:
                if w.word == wd:
                    w.add()
                    isIn = True
                    break
            if isIn == False:
                keywords.append(Word(wd, 1))
    return keywords
    
def tweet_sentiment(tags, tags_2):
    sentiment = 0
    for i in range(len(tags)):
        if tags[i] == "B-positive" or tags[i] == "I-positive":
            sentiment += tags_2[i]
        if tags[i] == "B-negative" or tags[i] == "I-negative":
            sentiment -= tags_2[i]
    return sentiment

def emis_prob(state, word, training_data, emis_dict):
    if (state, word) in emis_dict.keys():
        return emis_dict[(state, word)]
    else:
        count_emission = 0 # count the emission from state to word
        count_state = 1
        count_word = 0

        for tweet in training_data:
            for j in range(len(tweet)):
                if tweet[j].split(" ")[0] == word:
                    count_word += 1
                if tweet[j].split(" ")[1] == state:
                    count_state += 1
                    if tweet[j].split(" ")[0] == word:
                        count_emission += 1

        if count_word == 0:
            result =  float(1/count_state)
        else:
            result = float(count_emission/count_state)

        emis_dict[(state, word)] = result
        return result


# 'a' represents 'state1' and 'b' represents 'state2'
def transAB_prob(a, b, training_data, trans_dict):
    if (a,b) in trans_dict.keys():
        return trans_dict[(a,b)]
    else:
        countAB = 0
        countA = 0

        if a == 'start':
            countA = len(training_data)
            for tweet in training_data:
                if len(tweet[0].split(" ")) > 1 and tweet[0].split(" ")[1] == b:
                    countAB += 1
        elif b == 'stop':
            for tweet in training_data:
                for j in range(len(tweet)):
                    if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == a:
                        countA += 1
                        if j == len(tweet) - 1:
                            countAB += 1
        elif a == 'stop' or b == 'start':
            trans_dict[(a,b)] = 0
            return 0
        else:
            for tweet in training_data:
                for j in range(len(tweet)-1):
                    if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == a:
                        countA += 1
                        if tweet[j+1].split(" ")[1] == b:
                            countAB += 1

        result = float(countAB/countA)
        trans_dict[(a,b)] = result
        return result


# 'a' represents 'state1', 'b' represents 'state2' and 'c' represents 'state3'
def trans_prob_ABC(a, b, c, training_data, trans_dict):
    if (a,b,c) in trans_dict.keys():
        return trans_dict[(a,b,c)]
    else:
        countABC = 0
        countAB = 0

        if a == "start":
            for tweet in training_data:
                if len(tweet[0].split(" ")) > 1 and tweet[0].split(" ")[1] == b:
                    countAB += 1
                    if tweet[1].split(" ")[1] == c:
                        countABC += 1
        elif c == "stop":
            for tweet in training_data:
                for j in range(1,len(tweet)):
                    if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == b and tweet[j-1].split(" ")[1] == c:
                        countAB += 1
                        if j == len(tweet) - 1:
                            countABC += 1
        elif a == "stop" or b == "stop" or b == "start" or c == "start":
            trans_dict[(a,b,c)] = 0
            return 0
        else:
            for tweet in training_data:
                for j in range(1,len(tweet)-1):
                    if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == b and tweet[j-1].split(" ")[1] == a:
                        countAB += 1
                        if tweet[j+1].split(" ")[1] == c:
                            countABC += 1

        if countAB == 0:
            result = 0
        else:
            result = float(countABC/countAB)

        trans_dict[(a,b,c)] = result
        return result


def viterbip5_label(dev_datapath, training_datapath, filename, filename_2, os):
    if os == "W":
        outpath = dev_datapath.rsplit("\\",maxsplit=1)[0] + "\\" + filename
        outpath_2 = dev_datapath.rsplit("\\",maxsplit=1)[0] + "\\" + filename_2
    else:
        outpath = dev_datapath.rsplit("/",maxsplit=1)[0] + "/" + filename
        outpath_2 = dev_datapath.rsplit("/",maxsplit=1)[0] + "/" + filename_2

    outfile = open(outpath, 'w', encoding='utf8')
    outfile_2 = open(outpath_2, 'w', encoding='utf8')
    training_data = Data_processor(training_datapath).data
    dev_data = Data_processor(dev_datapath)
    total_tweets = len(dev_data.data)

    transAB_dict = {}
    transABC_dict = {}
    emis_dict = {}
    keywords = word_cloud(dev_data.data)
    
    for tweet in range(total_tweets):
        score_dict = {}
        opYseq = viterbi_trigram_end(dev_data.data[tweet], emis_dict, transAB_dict, transABC_dict, training_data, score_dict)
        
        tags = opYseq[0].split(" ")
        tags_2 = []
        for word in range(len(tags)):
            freq = 0
            for w in keywords:
                if w.word == dev_data.data[tweet][word]:
                    freq = w.freq
            tags_2.append(freq)
                    
            output = dev_data.data[tweet][word] + " " + tags[word] + " " + str(tags_2[word]) + "\n"
            outfile.write(output)
        
        sentiment = tweet_sentiment(tags, tags_2)
        if sentiment > 0:
            outfile_2.write("1" + "\n")
        elif sentiment < 0:
            outfile_2.write("-1" + "\n")
        else:
            outfile_2.write("0" + "\n")
        outfile.write("\n")
        print(str(tweet+1) + "/" + str(total_tweets) + " done")

    print("Labelling completed!")
    outfile.close()
    outfile_2.close()


def viterbi_trigram_start(sequence, state, emis_dict, trans_dict, training_data, score_dict):
    if (len(sequence), state) in score_dict.keys():
        return score_dict[(len(sequence), state)]
    else:
        score = transAB_prob("start", state, training_data, trans_dict) * emis_prob(state, sequence[-1], training_data, emis_dict)
        score_dict[(len(sequence), state)] = [(state, score)]
        return [(state, score)]


def viterbi_trigram_end(sequence, emis_dict, transAB_dict, transABC_dict, training_data, score_dict):
    max_y = ""
    max_score = 0

    for state in possible_states:
        if len(sequence) == 1:
            previous_max = viterbi_trigram_start(sequence, state, emis_dict, transAB_dict, training_data, score_dict)
            score = previous_max[0][1] * transAB_prob(state, "stop", training_data, transAB_dict)
            if score > max_score:
                max_y = previous_max[0][0]
                max_score = score
        else:
            previous_list = viterbi_trigram_recursive(sequence, state, emis_dict, transAB_dict, transABC_dict, training_data, score_dict)
            for j in previous_list:
                score = j[1] * (0.9 * transAB_prob(state, "stop", training_data, transAB_dict) + 0.1 * trans_prob_ABC(j[0].split(" ")[-2], j, "stop", training_data, transABC_dict))
                if score > max_score:
                    max_y = j[0]
                    max_score = score

    if max_y == "":
        previous_O_list = viterbi_trigram_recursive(sequence, "O", emis_dict, transAB_dict, transABC_dict, training_data, score_dict)
        max_y = previous_O_list[0][0]
        max_score = 0
    return (max_y, max_score)


def viterbi_trigram_recursive(sequence, state, emis_dict, transAB_dict, transABC_dict, training_data, score_dict):
    if (len(sequence), state) in score_dict.keys():
        return score_dict[(len(sequence), state)]
    else:
        state_list = []
        for prev_state in possible_states:
            if len(sequence) == 2:
                previous_list = viterbi_trigram_start(sequence[:-1], prev_state, emis_dict, transAB_dict, training_data, score_dict)
            else:
                previous_list = viterbi_trigram_recursive(sequence[:-1], prev_state, emis_dict, transAB_dict, transABC_dict, training_data, score_dict)

            max_y = ""
            max_score = 0

            for k in previous_list:
                if len(sequence) == 2:
                    max_score = k[1] * transAB_prob(prev_state, state, training_data, transAB_dict) * emis_prob(state, sequence[-1], training_data, emis_dict)
                    max_y = k[0] + " " + state
                else:
                    score = k[1] * (0.9 * transAB_prob(prev_state, state, training_data, transAB_dict) + 0.1 * trans_prob_ABC(k[0].split(" ")[-2], prev_state, state, training_data, transABC_dict)) * emis_prob(state, sequence[-1], training_data, emis_dict)
                    if score > max_score:
                        max_y = k[0] + " " + state
                        max_score = score

            if max_y == "":
                max_y = previous_list[0][0] + " " + state
            state_list.append((max_y, max_score))

        score_dict[(len(sequence), state)] = state_list
        return state_list

viterbip5_label("D:\MachineLearning\Project\dev.in", "train.txt", "dev.out", "dev_tweets.out", "W")
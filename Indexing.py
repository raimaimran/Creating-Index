import os
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json
import math
# global variable(s)
doc_count = 1

# ---------------------------------------------TASK 1: Preprocessing:---------------------------------------------------
whitelist = [
  'p', 'h1', 'h2', 'h3'

]


def generate_numbered_list(dir_name):
    global doc_count
    all_doc = os.listdir(dir_name)
    all_doc = filter(lambda x: x[-4:] != '.txt', all_doc)
    doc_id = {}
    for doc in sorted(all_doc):
        doc_id_val = doc_count
        doc_id[doc] = doc_id_val
        doc_count += 1

    with open(dir_name + "numbered_index.txt", 'w') as file:
        file.write(json.dumps(doc_id))


def save_index_to_file(dir_name , index):
    with open(dir_name + "temp_index.txt", 'w') as file:
        file.write(json.dumps(index))


def tokenizer(dir_name):
    token_pair = {}

    # reading numbered document from files
    with open(dir_name + 'numbered_index.txt') as f:
        js = f.read()
    js = json.loads(js)

    # listing all directories in dir_name
    all_doc = os.listdir(dir_name)
    txt_files = filter(lambda x: x[-4:] != '.txt', all_doc)  # reading non-text files

    # iterating one by one
    for doc in txt_files:
        output = ''
        path = os.path.join(dir_name, doc)
        fd = os.open(path, os.O_RDONLY)
        ret = os.read(fd, os.path.getsize(fd))
        os.close(fd)

        # parsing html file
        soup = BeautifulSoup(ret, 'html.parser')
        soup.find_all(text=True)
        text = [t for t in soup.find_all(text=True) if t.parent.name in whitelist]
        output += '{} '.format(text)

        # converting to lower case
        # splitting into tokens
        # removing punc/aplha numeric tokens
        #  = word_tokenize(output)
        tokens = word_tokenize(output)
        tokens = [w.lower() for w in tokens if w.isalpha()]
        # print("tokens", tokens)

        # removing stop words
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in tokens if not w in stop_words]
        # print(filtered_sentence)

        # applying stemming
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in filtered_sentence]
        # print(stemmed)

        for token in stemmed:
            posting = token_pair.get(token, None)
            if posting is None:
                posting = [js[doc]]

            else:
                if js[doc] not in posting:
                    posting.append(js[doc])
            token_pair[token] = posting
            # print(token_pair)

    print(token_pair)

    save_index_to_file(dir_name, token_pair)

# ------------------------------------------TASK 2: Creating Inverted Index:--------------------------------------------


def load_numbered_index(index_path):
    with open(index_path) as f:
        js = f.read()
    js = json.loads(js)
    return js


def load_temp_index(index_path):
    with open(index_path) as f:
        js = f.read()
    js = json.loads(js)
    return js


def load_temp_posting_list(temp_p_path):
    with open(temp_p_path) as f:
        js = f.read()
    js = json.loads(js)
    return js


def get_tokens(doc_int_id, dir_name):
    numbered_index = load_numbered_index(dir_name + 'numbered_index.txt')
    key_list = list(numbered_index.keys())
    val_list = list(numbered_index.values())

    # getting position in numbered list
    position = val_list.index(int(doc_int_id))
    doc_id = key_list[position]

    # print(doc_id)
    path = os.path.join(dir_name, doc_id)
    fd = os.open(path, os.O_RDONLY)
    ret = os.read(fd, os.path.getsize(fd))
    os.close(fd)

    # parsing html file
    output = ''
    soup = BeautifulSoup(ret, 'html.parser')
    soup.find_all(text=True)
    text = [t for t in soup.find_all(text=True) if t.parent.name in whitelist]
    output += '{} '.format(text)

    # converting to lower case, splitting into tokens, removing punc/aplha numeric tokens
    tokens = word_tokenize(output)
    tokens = [w.lower() for w in tokens if w.isalpha()]

    # removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stop_words]

    # applying stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_sentence]
    return stemmed


def inverted_index(dir_name):

    temp_index = load_temp_index(dir_name + "temp_index.txt")
    invert_index = {}

    # delta encoding
    pos_delta = {key: 0 for key in temp_index}

    numbered_index = load_numbered_index(dir_name + 'doc_info.txt')
    doc = numbered_index.keys()
    doc = [int(i) for i in doc]

    # iterating over all doc id's
    for doc_id in doc:
        print(doc_id)
        token_list = get_tokens(doc_id, dir_name)
        for pos, term in enumerate(token_list):
            # if terms exists
            if term in invert_index:
                # print(invert_index[term][1])
                # document found -> append position, else if not found -> set position
                if doc_id in invert_index[term][1]:
                    invert_index[term][1][doc_id].append((pos + 1) - pos_delta[term])
                    pos_delta[term] = pos + 1
                else:
                    invert_index[term][0] = invert_index[term][0] + 1
                    invert_index[term][1][doc_id] = [pos + 1]
                    pos_delta[term] = pos + 1

            # If term does not exist in the positional index dictionary (first encounter).
            else:
                invert_index[term] = []
                invert_index[term].append(1)
                # The postings list is initially empty.
                invert_index[term].append({})
                # Add doc ID to postings list.
                invert_index[term][1][doc_id] = [pos + 1]
                pos_delta[term] = pos + 1
                # print(invert_index)

    with open(dir_name + "temp_posting_list.txt", 'w') as file:
        file.write(json.dumps(invert_index))


def storing_indexes(dir_name):
    posting_list = load_temp_posting_list(dir_name + "temp_posting_list.txt")
    posting_list = dict(sorted(posting_list.items()))
    print(posting_list)
    terms = list(posting_list.keys())

    for term in terms:
        f = open(dir_name + "posting_list.txt", "a")
        i = open(dir_name + "index_terms.txt", "a")
        pos = f.tell()
        print(pos)

        index_dict = {}
        index_dict[term] = [pos]
        i.write(json.dumps(index_dict))
        i.write('\n')
        i.close()

        posting_list_final = {}
        posting_list_final[posting_list[term][0]] = posting_list[term][1]
        f.write(json.dumps(posting_list_final))
        f.write('\n')
        f.close()


# ----------------------------------------------TASK 3: Merging Indices:------------------------------------------------


def merge_inverted_indexes():
    t1 = open('corpus/corpus1/1index_terms.txt', 'r')
    t2 = open('corpus/corpus1/2index_terms.txt', 'r')
    t3 = open('corpus/corpus1/3index_terms.txt', 'r')

    p1 = open('corpus/corpus1/1posting_list.txt', 'r')
    p2 = open('corpus/corpus1/2posting_list.txt', 'r')
    p3 = open('corpus/corpus1/3posting_list.txt', 'r')

    content_1 = t1.readline()
    content_2 = t2.readline()
    content_3 = t3.readline()

    term1 = json.loads(content_1)
    term2 = json.loads(content_2)
    term3 = json.loads(content_3)

    # term1_list = list(term1.keys())
    # term2_list = list(term2.keys())
    # term3_list = list(term3.keys())

    buff_terms_1 = str(list(term1)[0])
    buff_terms_2 = str(list(term2)[0])
    buff_terms_3 = str(list(term3)[0])
    term = buff_terms_1
    p = p1
    while content_1 is not None or content_2 is not None or content_3 is not None:

        f = open("final_posting_list.txt", "a")
        i = open("final_index_terms.txt", "a")
        pos = f.tell()

        if buff_terms_1 <= buff_terms_2 and buff_terms_1 <= buff_terms_3:
            print(buff_terms_1)
            # print("in1")
            term = buff_terms_1
            content_1 = t1.readline()
            term1 = json.loads(content_1)
            term1_list = list(term1.keys())
            buff_terms_1 = term1_list[0]
            p = p1
        elif buff_terms_2 <= buff_terms_3 and buff_terms_2 <= buff_terms_1:
            print(buff_terms_2)
            # print('in2')
            term = buff_terms_2
            content_2 = t2.readline()
            term2 = json.loads(content_2)
            term2_list = list(term2.keys())
            buff_terms_2 = term2_list[0]
            p = p2
        elif buff_terms_3 <= buff_terms_2 and buff_terms_3 <= buff_terms_1:
            # print("in3")
            term = buff_terms_3
            print(buff_terms_3)
            content_3 = t3.readline()
            term3 = json.loads(content_3)
            term3_list = list(term3.keys())
            buff_terms_3 = term3_list[0]
            p = p3
        # print(term)
        index_dict = {}
        index_dict[term] = [pos]
        i.write(json.dumps(index_dict))
        i.write('\n')
        i.close()

        posting = p.readline()
        posting_list_final = json.loads(posting)
        print(posting_list_final)
        f.write(json.dumps(posting_list_final))
        f.write('\n')
        f.close()

    if content_1 is None:
        while content_2 is not None or content_3 is not None:
            f = open("final_posting_list.txt", "a")
            i = open("final_index_terms.txt", "a")
            pos = f.tell()

            if buff_terms_2 <= buff_terms_3:
                print(buff_terms_2)
                # print('in2')
                term = buff_terms_2
                content_2 = t2.readline()
                term2 = json.loads(content_2)
                term2_list = list(term2.keys())
                buff_terms_2 = term2_list[0]
                p = p2
            elif buff_terms_3 <= buff_terms_2:
                # print("in3")
                term = buff_terms_3
                print(buff_terms_3)
                content_3 = t3.readline()
                term3 = json.loads(content_3)
                term3_list = list(term3.keys())
                buff_terms_3 = term3_list[0]
                p = p3

            index_dict = {}
            index_dict[term] = [pos]
            i.write(json.dumps(index_dict))
            i.write('\n')
            i.close()

            posting = p.readline()
            posting_list_final = json.loads(posting)
            print(posting_list_final)
            f.write(json.dumps(posting_list_final))
            f.write('\n')
            f.close()

    elif content_2 is None:
        while content_1 is not None or content_3 is not None:
            f = open("final_posting_list.txt", "a")
            i = open("final_index_terms.txt", "a")
            pos = f.tell()

            if buff_terms_1 <= buff_terms_3:
                print(buff_terms_1)
                # print("in1")
                term = buff_terms_1
                content_1 = t1.readline()
                term1 = json.loads(content_1)
                term1_list = list(term1.keys())
                buff_terms_1 = term1_list[0]
                p = p1

            elif buff_terms_3 <= buff_terms_1:
                # print("in3")
                term = buff_terms_3
                print(buff_terms_3)
                content_3 = t3.readline()
                term3 = json.loads(content_3)
                term3_list = list(term3.keys())
                buff_terms_3 = term3_list[0]
                p = p3

            index_dict = {}
            index_dict[term] = [pos]
            i.write(json.dumps(index_dict))
            i.write('\n')
            i.close()

            posting = p.readline()
            posting_list_final = json.loads(posting)
            print(posting_list_final)
            f.write(json.dumps(posting_list_final))
            f.write('\n')
            f.close()

    elif content_3 is None:
        while content_2 is not None or content_1 is not None:
            f = open("final_posting_list.txt", "a")
            i = open("final_index_terms.txt", "a")
            pos = f.tell()

            if buff_terms_1 <= buff_terms_2:
                print(buff_terms_1)
                print("in1")
                term = buff_terms_1
                content_1 = t1.readline()
                term1 = json.loads(content_1)
                term1_list = list(term1.keys())
                buff_terms_1 = term1_list[0]
                p = p1
            elif buff_terms_2 <= buff_terms_1:
                print(buff_terms_2)
                print('in2')
                term = buff_terms_2
                content_2 = t2.readline()
                term2 = json.loads(content_2)
                term2_list = list(term2.keys())
                buff_terms_2 = term2_list[0]
                p = p2

            index_dict = {}
            index_dict[term] = [pos]
            i.write(json.dumps(index_dict))
            i.write('\n')
            i.close()

            posting = p.readline()
            posting_list_final = json.loads(posting)
            print(posting_list_final)
            f.write(json.dumps(posting_list_final))
            f.write('\n')
            f.close()

    if content_2 is None and content_3 is None:
        while content_1 is not None:
            f = open("final_posting_list.txt", "a")
            i = open("final_index_terms.txt", "a")
            pos = f.tell()

            term = buff_terms_1
            content_1 = t1.readline()
            term1 = json.loads(content_1)
            term1_list = list(term1.keys())
            buff_terms_1 = term1_list[0]
            p = p1

            index_dict = {}
            index_dict[term] = [pos]
            i.write(json.dumps(index_dict))
            i.write('\n')
            i.close()

            posting = p.readline()
            posting_list_final = json.loads(posting)
            print(posting_list_final)
            f.write(json.dumps(posting_list_final))
            f.write('\n')
            f.close()


# ------------------------------------------TASK 4: Additional Information:---------------------------------------------


def doc_length(doc_path):
    fd = os.open(doc_path, os.O_RDONLY)
    ret = os.read(fd, os.path.getsize(fd))
    os.close(fd)

    # parsing html file
    output = ''
    soup = BeautifulSoup(ret, 'html.parser')
    soup.find_all(text=True)
    text = [t for t in soup.find_all(text=True) if t.parent.name in whitelist]
    output += '{} '.format(text)

    return len(output)


def additional_information(dir_name):
    global doc_count
    all_doc = os.listdir(dir_name)
    all_doc = filter(lambda x: x[-4:] != '.txt', all_doc)
    doc_id = {}
    for doc in sorted(all_doc):
        doc_id_val = [dir_name + '/' + doc, doc_length(dir_name + '/' + doc), math.sqrt(doc_length(dir_name + '/' + doc)) ]
        doc_id[doc_count] = doc_id_val
        doc_count += 1
    # print(doc_id)

    with open(dir_name + "doc_info.txt", 'w') as file:
        file.write(json.dumps(doc_id))


# ---------------------------------------TASK 5: Reading Inverted Indices:----------------------------------------------

def read_inverted_index(terms):

    # preprocessing
    tokens = word_tokenize(terms)
    tokens = [w.lower() for w in tokens if w.isalpha()]

    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stop_words]

    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in filtered_sentence]
    print(tokens)

    for token in tokens:
        not_found = False
        token_found = False
        i_t = open("final_index_terms.txt", "r")
        p_l = open("final_posting_list.txt", "r")

        line_idx = i_t.readline()
        while line_idx is not None:
            term = list(json.loads(line_idx).keys())[0]
            pl_pos = list((json.loads(line_idx).values()))[0]
            # found word
            if term == token:
                token_found = True
                # print(pl_pos)
                p_l.seek(pl_pos[0])
                posting_list = p_l.readline()
                posting = json.loads(posting_list)
                freq = int(list(posting.keys())[0])

                term_keys = list(posting[str(freq)].keys())
                # print(term_keys[0])
                numbered_index = {}
                if 1 <= int(term_keys[0]) <= 1131:
                    numbered_index = load_numbered_index('corpus/corpus1/1doc_info.txt')
                    print(token, "found in folder 1 in documents:")
                elif 1132 <= int(term_keys[0]) <= 2236:
                    numbered_index = load_numbered_index('corpus/corpus1/2doc_info.txt')
                    print(token, "found in folder 2 in documents:")
                elif 2237 <= int(term_keys[0]) <= 3465:
                    numbered_index = load_numbered_index('corpus/corpus1/3doc_info.txt')
                    print(token, "found in folder 3 in documents:")

                for key in term_keys:
                    # print(numbered_index)
                    print(numbered_index[str(key)][0])

            elif term > token:
                break

            line_idx = i_t.readline()

        if token_found is False:
            print(token, "not found")


# -------------------------------------------------------Main-----------------------------------------------------------
# code for task 1
# generates a numbered doc list for the corpus
# generate_numbered_list('corpus/corpus1/1')
# generate_numbered_list('corpus/corpus1/2')
# generate_numbered_list('corpus/corpus1/3')

# tokenizer('corpus/corpus1/1')
# tokenizer('corpus/corpus1/2')
# tokenizer('corpus/corpus1/3')

# code for task 4
# additional_information('corpus/corpus1/1')
# additional_information('corpus/corpus1/2')
# additional_information('corpus/corpus1/3')


# code for task 2
# inverted_index('corpus/corpus1/1')
# inverted_index('corpus/corpus1/2')
# inverted_index('corpus/corpus1/3')

# storing_indexes('corpus/corpus1/1')
# storing_indexes('corpus/corpus1/2')
# storing_indexes('corpus/corpus1/3')

# code for task 3
# merge_inverted_indexes()

# code for task 5
read_inverted_index("information retrieval iksp")


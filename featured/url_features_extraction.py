from urllib.parse import urlparse
from string import ascii_lowercase
import csv
import re
from xml.dom import minidom
import urllib
import pygeoip
import urllib
import urllib.request
import bs4
import sys
import requests
from bs4 import BeautifulSoup as bs
import math
import string
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import collections
from scipy.stats import ks_2samp
from scipy import stats
import tldextract
from posixpath import basename, dirname


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)

        if parts[0] == path:
            allparts.insert(1, parts[0])
            break
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])

    return allparts


def getTokens(url):
    return re.split('\W+', url)


def token_count(url):
    return len(getTokens(url))


def get_num_label(label):
    if label == "good":
        num_label = 1

    else:

        num_label = 0

    return num_label


def get_unpunctuated_url(url):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove punctuation from the url
    unpunctuated_url = ""
    for char in url:
        if char not in punctuations:
            unpunctuated_url = unpunctuated_url + char

    # return unpunctuated url
    return unpunctuated_url

#get english letter freq based https://en.wikipedia.org/wiki/Letter_frequency
def get_letter_freq(url):
    english_letter_freq = [('a', 8.17), ('b', 1.49), ('c', 2.78), ('d', 4.25), ('e', 12.70), ('f', 2.23), ('g', 2.02),
                           ('h', 6.09), ('i', 6.97), ('j', 0.15), ('k', 0.77), ('l', 4.03), ('m', 2.41), ('n', 6.75),

                           ('o', 7.51), ('p', 1.93), ('q', 0.10), ('r', 5.99), ('s', 6.33), ('t', 9.06), ('u', 2.76),
                           ('v', 0.98), ('w', 2.36), ('x', 0.15), ('y', 1.97), ('z', 0.07)]

    url_length = float(len(url))

    url_dictionary = collections.defaultdict(int)

    # the absolute frequencies
    for char in url:
        url_dictionary[char] += 1

    # the relative frequencies
    url_letter = []

    url_letter_freq = []

    for char in ascii_lowercase:

        if url_length > 0:

            url_letter.append((char, (url_dictionary[char] / url_length)))

            url_letter_freq.append([(char), round((url_dictionary[char] / url_length), 2)])

        else:

            url_length = 1

            url_letter.append((char, (url_dictionary[char] / url_length)))

            url_letter_freq.append([(char), round((url_dictionary[char] / url_length), 2)])

    url_letter_freq_list = []
    english_letter_freq_list = []

    i = 0

    while i < len(url_letter_freq):
        url_letter_freq_extract = url_letter_freq[i][1]
        english_letter_freq_extract = english_letter_freq[i][1]

        url_letter_freq_list.append(url_letter_freq_extract)
        english_letter_freq_list.append(english_letter_freq_extract)

        i += 1

    return url_letter_freq_list, english_letter_freq_list


"""--------------------------1-------------------------"""


def get_url_length(url):
    return len(url)


"""--------------------------2-------------------------"""


def get_path_length(path):
    return len(path)


"""--------------------------3-------------------------"""


def get_entropy(url, base=2.0):
    # make set with all unrepeatable characters from url
    url_dictionary = dict.fromkeys(list(url))

    # calculate character frequencies
    char_frequency = [float(url.count(char)) / len(url) for char in url_dictionary]

    # calculate Entropy
    entropy = -sum([ch * math.log(ch) / math.log(base) for ch in char_frequency])
    return entropy


"""--------------------------4-------------------------"""


def get_num_at(url):
    return url.count('@')


"""--------------------------5-------------------------"""


def get_num_dotcom(url):
    return url.count('.com')


"""--------------------------6-------------------------"""


def get_punctuation_count(url):
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

    punctuation_count = count(url, string.punctuation)

    return punctuation_count


"""--------------------------7-------------------------"""


def get_suspicious_word_count(token):
    suspicious_word_count = 0

    for word in suspicious_word_list:

        if word in token:
            suspicious_word_count += 1

    return suspicious_word_count

"""--------------------------8-------------------------"""

def check_if_IP(url):

    if (re.search(
            r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
            r'\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', url)) is not None:
        return 1  # IP present
    else:
        return 0  # IP not present

"""--------------------------9-------------------------"""


def get_elucidean_distance(url):
    url_list = list(url)

    url_list_ord = []
    for i in url_list:
        url_list_ord.append(int(ord(i)))

    j = 0

    sum_df_sq = 0

    while j < len(url_list_ord) - 1:
        df_sq = (abs(url_list_ord[j] - url_list_ord[j + 1])) ** 2

        sum_df_sq += df_sq

        j += 1

    df_sq2 = (abs(url_list_ord[0] - url_list_ord[len(url_list) - 1])) ** 2

    sum_df_sq += df_sq2

    return sum_df_sq


"""--------------------------10 - 11 (p_value)------------------------"""


def get_ks_stats(url_letter_freq_list, english_letter_freq_list):
    ks_statistic, p_value = ks_2samp(url_letter_freq_list, english_letter_freq_list)

    return ks_statistic, p_value


"""--------------------------12-------------------------"""


def get_kl_divergence(url_letter_freq_list, english_letter_freq_list):
    kl_divergence = stats.entropy(pk=url_letter_freq_list, qk=english_letter_freq_list)

    return kl_divergence


"""--------------------------13-------------------------"""


# domain_name includes primary domain and subdomain

def get_domain_name(url):
    domain = tldextract.extract(url).domain
    subdomain = tldextract.extract(url).subdomain

    if len(subdomain) == 0:
        domain_name = domain
    else:
        domain_name = subdomain + "." + domain

    return (domain_name)


def get_domain_name_token(url):
    domain = tldextract.extract(url).domain
    subdomain = tldextract.extract(url).subdomain

    if len(subdomain) == 0:
        domain_name = domain
    else:
        domain_name = subdomain + "." + domain

    return re.split('\W+', domain_name)


def get_domain_name_token_count(url):
    domain_name_token = get_domain_name_token(url)

    return len(domain_name_token)


"""--------------------------14-------------------------"""


def get_average_domain_name_token_length(domain_name_token):
    total_domain_name_token_length = 0

    for i in domain_name_token:
        total_domain_name_token_length += len(i)

    return total_domain_name_token_length / len(domain_name_token)


"""--------------------------15-------------------------"""


def get_path(path):
    if path == '':
        return 0
    else:
        return path


def get_path_token(path):
    if path == '':
        return 0
    else:
        return re.split('\W+', path)


def get_path_token_count(path):

    if get_path_token(path) == 0:
        return 0
    else:
        path_token_count = len(get_path_token(path))
        for i in get_path_token(path):
            if i == '/' or i == '':
                path_token_count -= 1
        return path_token_count


"""--------------------------16-------------------------"""


def get_average_path_token_length(path):
    total_path_token_length = 0
    if get_path_token(path) == 0:
        return 0
    else:
        path_token_count = len(get_path_token(path))
        for i in get_path_token(path):
            if i == '/' or i == '':
                path_token_count -= 1
                total_path_token_length += 0
            else:
                total_path_token_length += len(i)
        if path_token_count == 0:
            return 0
        else:
            return total_path_token_length / path_token_count


"""--------------------------17-------------------------"""


def get_longest_domain_token_length(domain_name_token):
    domain_token_length = []

    for i in domain_name_token:
        domain_token_length.append(len(i))

    return max(domain_token_length)


"""--------------------------18-------------------------"""


def get_longest_path_token_length(path):
    path_token_length = []

    if get_path_token(path) == 0:
        return 0
    else:
        path_token_count = len(get_path_token(path))
        for i in get_path_token(path):
            if i == '/' or i == '':
                path_token_count -= 1

            else:
                path_token_length.append(len(i))
        if path_token_count == 0:
            return 0
        else:
            return max(path_token_length)


"""--------------------------19-------------------------"""


def get_num_dot(url):
    return url.count('.')


"""--------------------------20-------------------------"""


def if_exist_port_number(url):
    o = urlparse(url)
    if o.port != None:
        return 1
    return 0


"""--------------------------21-------------------------"""


def get_hyphen_count(url):
    return url.count('-')


"""--------------------------22------------------------"""


def get_directory_length(path):
    directory = dirname(path)
    return len(directory[1:])


"""--------------------------23------------------------"""


def get_sub_directory(url):
    parse_object = urlparse(url)
    path = parse_object.path
    dirname_path = dirname(path)
    path = path.replace(dirname_path, '')

    if path == url:

        return ''

    else:

        return path


def get_sub_directory_special_count(sub_directory):
    count = 0

    delim = ['-', '.', '_', '~', ':', '/', '?', '#', '[', ']', '@', '!', '$', '&', '(', ')', '*', '+', ',', ';', '=',
             '`']
    sub_path = sub_directory.strip("/")

    for lim in delim:
        if lim in sub_path:
            count = count + sub_path.count(lim)

    return count


"""--------------------------24------------------------"""


def get_sub_directory_tokens_count(url):
    path = get_sub_directory(url)

    tokens = re.split('\W+', path)
    tokens = filter(None, tokens)
    return len(list(tokens))


"""--------------------------25------------------------"""


def get_max_length_sub_dir_token(url):
    sub_dir_token_length = []

    path = get_sub_directory(url)

    if path == '' or path == '/':
        return 0
    else:
        tokens = re.split('\W+', path)
        tokens = filter(None, tokens)

        for i in list(tokens):
            sub_dir_token_length.append(len(i))

        if len(sub_dir_token_length) > 0:
            return max(sub_dir_token_length)

        else:

            return 0


"""--------------------------26------------------------"""


def get_max_dot_sub_dir_token(url):
    sub_dir_token_dot_count = []

    token_dot_count = 0

    path = get_sub_directory(url)

    if path == '' or path == '/':
        return 0
    else:
        tokens = re.split('\W+', path)
        tokens = filter(None, tokens)

        for i in list(tokens):

            for j in i:

                if j == '.':
                    token_dot_count += 1

                sub_dir_token_dot_count.append(token_dot_count)

        if len(sub_dir_token_dot_count) > 0:

            return max(sub_dir_token_dot_count)

        else:

            return 0


"""--------------------------27------------------------"""


def get_filename(url, domain):
    parse_object = urlparse(url)
    filename = basename(parse_object.path)

    if (filename.split('.')[0]) == domain:

        return ''

    else:

        return (filename.split('.')[0])


def get_filename_length(filename):
    return len(filename)


"""--------------------------28------------------------"""


def get_num_dot_filename(filename):
    return filename.count('.')


"""--------------------------29------------------------"""


def get_num_delim_filename(filename):
    count = 0

    delim = ['-', '.', '_', '~', ':', '/', '?', '#', '[', ']', '@', '!', '$', '&', '(', ')', '*', '+', ',', ';', '=',
             '`']

    for lim in delim:
        if lim in filename:
            count = count + filename.count(lim)
    return count


"""--------------------------30------------------------"""


def get_argument(url):
    return urlparse(url).query


def get_argument_length(url):
    return len(urlparse(url).query)


"""--------------------------31------------------------"""


def get_num_delim_argument(url):
    count = 0

    delim = ['-', '.', '_', '~', ':', '/', '?', '#', '[', ']', '@', '!', '$', '&', '(', ')', '*', '+', ',', ';', '=',
             '`']

    argument = urlparse(url).query

    for lim in delim:

        if lim in argument:
            count = count + argument.count(lim)
    return count

"""--------------------------32------------------------"""


def is_url_short(url):
    if ".ly/" in url:
        return 1  # short_url present
    else:
        return 0  # short_url not present


"""--------------------------33------------------------"""

def is_exe(url):
    if "exe" in url:
        return 1  # 'exe' present in url
    else:
        return 0  # 'exe' not present in url

# ----------------------------------------------------------------------------------------------------------------

output_file = open('url_features_extracted.csv', 'a', encoding='utf8')

writer = csv.writer(output_file, lineterminator='\r')

writer.writerow(["url", "url_length", "path_length", "entropy", "num_at",
                 "num_dotcom", "punctuation_count", "suspicious_word_count", "is_ip",
                 "elucidean_dist", "KS_stats", "p_value", "KL_diverg",
                 "dom_name_tok_cnt", "ave_dom_name_tok_len", "longest_path_token_length",
                 "path_tok_cnt", "ave_path_tok_len", "longest_path_tok_len",
                 "num_dot", "if_exist_port_num", "hyphen_count", "dir_len",
                 "sub_dir_special_cnt", "sub_dir_tok_cnt",
                 "max_len_sub_dir_tok", "max_dot_sub_dir_tok", "filename_len",
                 "num_dot_filename", "num_delim_filename", "arg_len",
                 "num_delim_arg",
                 "short_url_present", "exe_present", "num_label", "class_label"])

input_file = open('url_data.csv', encoding="utf8")

input_csv = csv.reader(input_file)

file = open('suspicious_word_list.csv', 'r', encoding='utf8')
suspicious_word_list_csv = csv.reader(file)

suspicious_word_list = []
for word in suspicious_word_list_csv:
    if type(word[0]) != str:
        word = unicode_decode(word[0])
        sus_word = word
    sus_word = word[0]
    suspicious_word_list.append(sus_word)

dataline = 1

for item in input_csv:

    url = item[0]

    token = re.split('\W+|_', item[0])  # regular expression split

    if item[0][:8] != "https://" and item[0][:7] != "http://":
        item[0] = "http://" + item[0]

    url_info = urlparse(item[0])
    path = url_info.path
    host = url_info.netloc

    """--------------------------1-------------------------"""

    url_length = get_url_length(item[0])  # url size

    # print("url_length = ", url_length)

    """--------------------------2-------------------------"""

    path_length = get_path_length(path)  # path length

    # print("path_length = ", path_length)

    """--------------------------3-------------------------"""

    entropy = get_entropy(item[0])

    # print("entropy = ", entropy)

    """--------------------------4-------------------------"""

    num_at = get_num_at(item[0])

    # print("num_at = ", num_at)

    """--------------------------5-------------------------"""

    num_dotcom = get_num_dotcom(item[0])

    # print("num_dotcom = ", num_dotcom)

    """--------------------------6-------------------------"""

    punctuation_count = get_punctuation_count(item[0])

    # print("punctuation_count = ", punctuation_count)

    """--------------------------7-------------------------"""

    suspicious_word_count = get_suspicious_word_count(token)

    # print("suspicious_word_count = ", suspicious_word_count)

    """--------------------------8-------------------------"""

    is_ip = check_if_IP(item[0])

    # print("is_ip = ", is_ip)

    """--------------------------9-------------------------"""

    elucidean_distance = get_elucidean_distance(item[0])

    # print("elucidean distance = ", elucidean_distance[0][0])

    """--------------------------10 - 11-------------------------"""

    unpunctuated_url = get_unpunctuated_url(item[0])

    url_letter_freq_list, english_letter_freq_list = get_letter_freq(unpunctuated_url)

    ks_statistic, p_value = get_ks_stats(url_letter_freq_list, english_letter_freq_list)

    # print("ks_statistic = ", ks_statistic)
    # print("p_value = ", p_value)

    """--------------------------12-------------------------"""

    kl_divergence = get_kl_divergence(url_letter_freq_list, english_letter_freq_list)

    # print("kl_divergence = ", kl_divergence)

    """--------------------------13-------------------------"""

    # domain_name = get_domain_name(url)

    domain_name_token_count = get_domain_name_token_count(url)

    """--------------------------14-------------------------"""

    average_domain_name_token_length = get_average_domain_name_token_length(get_domain_name_token(url))

    """--------------------------15-------------------------"""

    # path2 = get_path(path)

    path_token_count = get_path_token_count(path)

    """--------------------------16-------------------------"""

    average_path_token_length = get_average_path_token_length(path)

    """--------------------------17-------------------------"""

    longest_domain_token_length = get_longest_domain_token_length(get_domain_name_token(url))

    """--------------------------18-------------------------"""

    longest_path_token_length = get_longest_path_token_length(path)

    """--------------------------19-------------------------"""

    num_dot = get_num_dot(url)

    """--------------------------20-------------------------"""

    if_exist_port_num = if_exist_port_number(url)  # not verified

    """--------------------------21-------------------------"""

    hyphen_count = get_hyphen_count(url)

    """--------------------------22-------------------------"""

    # directory = dirname(path)

    directory_length = get_directory_length(path)

    """--------------------------23-------------------------"""

    sub_directory = get_sub_directory(url)

    sub_directory_special_count = get_sub_directory_special_count(sub_directory)

    """--------------------------24-------------------------"""

    sub_directory_tokens_count = get_sub_directory_tokens_count(url)

    """--------------------------25-------------------------"""

    max_length_sub_dir_token = get_max_length_sub_dir_token(url)

    """--------------------------26-------------------------"""

    max_dot_sub_dir_token = get_max_dot_sub_dir_token(url)  # not verified

    """--------------------------27-------------------------"""

    filename = get_filename(url, tldextract.extract(url).domain)

    filename_length = get_filename_length(filename)

    """--------------------------28-------------------------"""

    num_dot_filename = get_num_dot_filename(filename)

    """--------------------------29-------------------------"""

    num_delim_filename = get_num_delim_filename(filename)

    """--------------------------30-------------------------"""

    argument = urlparse(url).query

    argument_length = get_argument_length(url)

    """--------------------------31-------------------------"""

    num_delim_argument = get_num_delim_argument(url)

    """--------------------------32------------------------"""

    short_url_present = is_url_short(url)

    """--------------------------33------------------------"""

    exe_present = is_exe(url)

    """--------------------------last feature-------------------------"""

    class_label = item[1]

    num_label = get_num_label(item[1])

    writer.writerow([item[0], url_length, path_length, entropy, num_at, num_dotcom, punctuation_count,
                     suspicious_word_count, is_ip, elucidean_distance, ks_statistic, p_value, kl_divergence,
                     domain_name_token_count, average_domain_name_token_length,
                     path_token_count, average_path_token_length, longest_domain_token_length, longest_path_token_length,
                     num_dot, if_exist_port_num, hyphen_count, directory_length,
                     sub_directory_special_count, sub_directory_tokens_count,
                     max_length_sub_dir_token, max_dot_sub_dir_token, filename_length,
                     num_dot_filename, num_delim_filename, argument_length,
                     num_delim_argument, short_url_present, exe_present, num_label, class_label])

    print('writing dataline # %s' % dataline)

    dataline += 1

print("url features extraction completed !!!")
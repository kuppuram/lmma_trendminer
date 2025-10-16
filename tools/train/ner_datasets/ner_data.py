def get_ner_data():
    return [
        {'text': 'review trends from june to aug 2015 with a 4 score', 'labels': 'O O O B-MONTH I-MONTH I-MONTH B-YEAR O O B-SCORE O'},
        {'text': 'trends for feb 2018 with 3 score', 'labels': 'O O B-MONTH B-YEAR O B-SCORE O'},
        {'text': 'product reviews for this year with 5 score', 'labels': 'O O O B-DATE I-DATE O B-SCORE O'},
        {'text': 'analyze reviews from sept 2012 to dec 2013', 'labels': 'O O O B-MONTH B-YEAR O B-MONTH B-YEAR'},
        {'text': 'show trends from march to may with a 2 score', 'labels': 'O O O B-MONTH I-MONTH I-MONTH O O B-SCORE O'},
        {'text': 'what are trends for nov 2016 with score 1', 'labels': 'O O O O B-MONTH B-YEAR O O B-SCORE'},
        {'text': 'trends for the last quarter of 2019 with 4 score', 'labels': 'O O O B-DATE I-DATE I-DATE B-YEAR O B-SCORE O'},
        {'text': 'reviews from jan to march 2014', 'labels': 'O O B-MONTH I-MONTH I-MONTH B-YEAR'},
        {'text': 'product trends this month with score 5', 'labels': 'O O B-DATE I-DATE O O B-SCORE'},
        {'text': 'analyze trends from july 2011 to oct 2012 with score 3', 'labels': 'O O O B-MONTH B-YEAR O B-MONTH B-YEAR O O B-SCORE'}
    ]
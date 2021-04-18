import pandas as pd
from series2graph import *



class S2gWrapper :

    def __init__(self,df:pd.DataFrame,query_length,pattern_length) :
        self.df = df;
        self.query_length = query_length;
        self.pattern_length = pattern_length;



    def calc(self,index):
        s2g = Series2Graph(pattern_length=self.pattern_length)
        s2g.fit(self.df,index)
        s2g.score(self.query_length)
        return(s2g.all_score)


    
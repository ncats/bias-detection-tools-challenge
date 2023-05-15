##measure_disparity_methods.py for iAdeptive

import pandas as pd
from plotnine import *
from sklearn.metrics import roc_curve
from os import makedirs

class measured:
    fullnames = { #dictionary of full length names for each of the 12 metric functions
        'STP': 'Statistical Parity',
        'TPR': 'Equal Opportunity',
        'PPV': 'Predictive Parity',
        'FPR': 'Predictive Equality',
        'ACC': 'Accuracy',
        'TNR': 'True Negative Rate',
        'NPV': 'Negative Predictive Value',
        'FNR': 'False Negative Rate',
        'FDR': 'False Discovery Rate',
        'FOR': 'False Omission Rate',
        'TS': 'Threat Score',
        'FS': 'F1 Score'
    }
    
    def __init__(self, 
                 dataset,
                 democols,
                 intercols,
                 actualcol='actual',
                 predictedcol='predicted',
                 probabilitycol='probability',
                 weightscol='weights'):
        """
        Inputs
        dataset <class 'pandas.core.frame.DataFrame'>: data generated from a model you which to analyze
        democols <class 'list'>: list of strings of demographic column names
        intercols <class 'list'>: list of lists with pairs of strings of demo column names you wish to see interactions 
                                  between
        actualcol <class 'str'>: string of column name with actual values formatted as 0 or 1
        predictedcol <class 'str'>: string of column name with predicted values formatted as 0 or 1
        probabilitycol <class 'str'>: string of column name with probability values formatted as floats 0 to 1
        weightscol <class 'str'>: string of column name with weights formatted as ints or floats

        Output
        measured <class 'measure_disparity.measured'>: object of the class from measure_disparity.py
        """

        #read in inputs
        self.df = dataset.copy()
        self.democols = democols
        self.actcol = actualcol
        self.predcol = predictedcol
        self.probcol = probabilitycol
        self.wcol = weightscol

        self.tpcount = {} #dictionary for true positive count
        self.fpcount = {}
        self.fncount = {}
        self.tncount = {}

        #create column for subgroup interactions if intercols is not an empty list
        if intercols:
            self._intergroups(intercols)

        #create column with truth values
        self.df['truths'] = self.df[self.actcol] + (self.df[self.predcol] * 2)

        #calculating truth counts per demographic
        self._calc()

        #dictionary of shorthand names for each of the 12 metric functions
        self.metricnames = {
            'STP': self.StatParity,
            'TPR': self.EqualOpp,
            'PPV': self.PredParity,
            'FPR': self.PredEqual,
            'ACC': self.Accuracy,
            'TNR': self.TrueNeg,
            'NPV': self.NegPV,
            'FNR': self.FalseNeg,
            'FDR': self.FalseDis,
            'FOR': self.FalseOm,
            'TS': self.ThreatScore,
            'FS': self.FScore
        }
    
    def _calc(self):
        """
        Calculating truth counts for each demographic
        """
        for col in self.df:
            if col in self.democols:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    adf = self.df[self.df[col] == key]
                    if self.wcol in adf.columns:
                        tpdf = adf[adf.truths == 3]
                        fpdf = adf[adf.truths == 2]
                        fndf = adf[adf.truths == 1]
                        tndf = adf[adf.truths == 0]
                        self.tpcount[key] = sum(tpdf.weights)
                        self.fpcount[key] = sum(fpdf.weights)
                        self.fncount[key] = sum(fndf.weights)
                        self.tncount[key] = sum(tndf.weights)
                    else:
                        self.tpcount[key] = sum(adf.truths == 3)
                        self.fpcount[key] = sum(adf.truths == 2)
                        self.fncount[key] = sum(adf.truths == 1)
                        self.tncount[key] = sum(adf.truths == 0)
    
    def _intergroups(self, intercols):
        """
        Create new intersectional columns with pairs of subgroups
        """
        for pair in intercols:
            pairname = pair[0] + '-' + pair[1]
            self.df[pairname] = self.df[pair[0]] + '-' + self.df[pair[1]]
            self.democols.append(pairname) #adding in intersectional columns

    def _todf(self, adict, shortname):
        """
        Converting the output to a dataframe format
        """
        adf = pd.DataFrame.from_dict(adict, orient='index', columns=[self.fullnames[shortname]])
        adf.reset_index(inplace=True)
        adf = adf.rename(columns={'index': 'Subgroup'})
        return adf
    
    @staticmethod
    def _printtable(mname, metricdict):
        """
        Printing out a table with the requested metrics
        """
        dlen = len('Subgroup') + 1
        for key in metricdict:
            if len(key) > dlen:
                dlen = len(key) + 1 #setting length of first column for the table
        print('{0:<{1}}|{2:>{3}}'.format('Subgroup',dlen,mname, len(mname)))
        for item in metricdict:
            print('{0:<{1}}|{2:>{3}}'.format(item,dlen,metricdict[item], len(mname)))
    
    def MetricPlots(self, 
                    colname, 
                    privileged, 
                    draw=True, 
                    metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
                    graphpath=None):
        """
        Outputting a figure with graphs of all the metrics for a subgroup category

        Inputs
        colname <class 'str'>: string of demographic column name
        privileged <class 'str'>: string of name for the privileged subgroup within this demographic column
        draw <class 'bool'>: boolean of whether to draw the plot or not
        metrics <class 'list'>: list of strings of shorthand names of metrics to make graphs of
        graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the
                                 plot will not be saved
                            
        Output
        aplot <class 'plotnine.ggplot.ggplot'>: plotnine plot of the metrics and demographics chosen
        """
        metricsdf = pd.DataFrame()
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](colname)
                else:
                    metricsdf[self.fullnames[name]] = self.metricnames[name](colname).iloc[:,1]
        demokeys = self.df[colname].unique().tolist()
        metricsdf = metricsdf.set_index('Subgroup')
        metricsdf.index.name = None
        metricsdf = metricsdf.loc[demokeys, :]
        metricsdf = metricsdf.transpose()
        demokeys.remove(privileged)
        for demo in demokeys:
            metricsdf[demo] = metricsdf[demo] / metricsdf[privileged] - 1
        metricsdf.reset_index(inplace=True)
        gheight = len(demokeys) * len(metrics) * 0.5
        if len(demokeys) > 1:
            metricsdf = pd.melt(metricsdf, id_vars='index', value_vars=demokeys)
            atitle = 'Fairness Metrics by ' + colname
            aplot = (ggplot(metricsdf)
            + aes(x='index', 
                  y='value', 
                  fill='index')
            + geom_bar(stat='identity', 
                       position='stack', 
                       show_legend=False)
            + scale_y_continuous(limits= (-0.6, 0.6), 
                                 breaks= [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6], 
                                 labels = (0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6))
            + theme(figure_size= (12, gheight))
            + coord_flip()
            + geom_hline(yintercept=0, linetype='dashed', alpha=0.5)
            + facet_wrap('variable', 
                         nrow=len(demokeys))
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[-0.2,-0.6,-0.6,-0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,-0.2,-0.2,0.2], 
                       fill='#98FB98', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,0.6,0.6,0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + labs(y='Score Ratio', 
                   x='Metric', 
                   title=atitle)
            )
        else: #when only two demographics to compare
            atitle = 'Fairness Metrics ' + demokeys[0] + ' vs ' + privileged
            aplot = (ggplot(metricsdf)
            + aes(x='index', 
                  y=demokeys[0])
            + geom_bar(stat='identity', 
                       position='stack')
            + scale_y_continuous(limits= (-0.6, 0.6), 
                                 breaks= [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6], 
                                 labels = (0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6))
            + theme(figure_size = (12, gheight))
            + coord_flip()
            + geom_hline(yintercept=0, linetype='dashed', alpha=0.5)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[-0.2,-0.6,-0.6,-0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,-0.2,-0.2,0.2], 
                       fill='#98FB98', 
                       alpha=0.2)
            + annotate('polygon', 
                       x=[0,0,13,13], 
                       y=[0.2,0.6,0.6,0.2], 
                       fill='#D9544D', 
                       alpha=0.2)
            + labs(y='Score Ratio', 
                   x='Metric', 
                   title=atitle)
            )
        if draw:
            print(aplot)
        if graphpath != None:
            makedirs(graphpath, exist_ok=True)
            fname = 'FairnessMetrics'
            for item in metrics:
                fname += item
            fname += 'by' + colname + '.png'
            aplot.save(filename=fname, path=graphpath)
        return aplot
    
    def RocPlots(self, 
                 colname, 
                 draw=True, 
                 graphpath=None):
        """
        Outputting the receiver operating characteristic curves for one category of subgroups

        Inputs
        colname <class 'str'>: string of demographic column name
        draw <class 'bool'>: boolean of whether to draw the graphs or not
        graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the 
                                 plot will not be saved
        
        Outputs
        aplot <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve
        rocgraph2 <class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve zoomed in on the upper left hand
                                                quadrant
        """
        rocdf = pd.DataFrame(columns=['subgroup', 'fpr', 'tpr'])
        for subgroup in self.df[colname].unique():
            adf = self.df[self.df[colname] == subgroup]
            fpr, tpr, thresholds = roc_curve(adf[self.actcol], adf[self.probcol])
            rdf = pd.DataFrame()
            rdf['fpr'] = fpr
            rdf['tpr'] = tpr
            rdf['subgroup'] = subgroup
            rocdf = pd.concat([rocdf, rdf])
        atitle = 'Receiver Operating Characteristic Curves by ' + colname
        aplot = (ggplot(rocdf)
        + aes(x='fpr', 
              y='tpr', 
              color='subgroup')
        + geom_line()
        + xlim(0, 1)
        + ylim(0, 1)
        + labs(y='True Positive Rate', 
               x='False Positive Rate', 
               title=atitle)
        )
        atitle = 'Zoomed In ROC Curves by ' + colname
        rocgraph2 = (ggplot(rocdf)
        + aes(x='fpr', 
              y='tpr', 
              color='subgroup')
        + geom_line()
        + xlim(0, 0.5)
        + ylim(0.5, 1)
        + labs(y='True Positive Rate', 
               x='False Positive Rate', 
               title=atitle)
        )
        if draw:
            print(aplot)
            print(rocgraph2)
        if graphpath != None:
            makedirs(graphpath, exist_ok=True)
            fname = 'ROCcurvesby' + colname + '.png'
            aplot.save(filename=fname, path=graphpath)
            fname = 'ZoomedInROCcurvesby' + colname + '.png'
            rocgraph2.save(filename=fname, path=graphpath)
        return [aplot, rocgraph2]
    
    def PrintMetrics(self, 
                     columnlist=[], 
                     metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC']):
        """
        Printing out all the chosen metrics in a table

        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to print metrics for
        metrics <class 'list'>: list of shorthand names of the metrics to print out
        """
        metricsdf = pd.DataFrame()
        if not columnlist:
            columnlist = self.democols
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](columnlist)
                else:
                    adf = self.metricnames[name](columnlist)
                    metricsdf[self.fullnames[name]] = adf.iloc[:,1]
        megalist = []
        for col in metricsdf.columns:
            megalist.append(metricsdf[col].array.tolist())
        megalist = list(map(list, zip(*megalist)))
        demolen = 0
        for item in metricsdf['Subgroup']: #setting length of 1st column to longest subgroup name
            if len(item) > demolen:
                demolen = len(item)
        col_widths = [demolen]
        metricsdf = metricsdf.set_index('Subgroup') #moving Subgroups column back to the index
        for col in metricsdf.columns:
            col_widths.append(len(col))
        formatted = ' '.join(['%%%ds |' % (width + 1) for width in col_widths ])[:-1]
        metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
        colnames = metricsdf.columns

        #printing out the table
        print(formatted % tuple(colnames))
        for row in megalist:
            print(formatted % tuple(row))
    
    def PrintRatios(self, 
                    colname, 
                    privileged, 
                    metrics=['STP', 'TPR', 'PPV', 'FPR', 'ACC'], 
                    printout=True):
        """
        Printing out all the metrics as ratios for one demographic column

        Inputs
        colname <class 'str'>: string of demographic column name
        privileged <class 'str'>: string of name for the privileged subgroup within this demographic column
        metrics <class 'list'>: list of strings of shorthand names of metrics to calculate ratios for
        printout <class 'bool'>: boolean of whether or not to print out the table of ratios calculated

        Outputs
        metricsdf <class 'pandas.core.frame.DataFrame'>: table of the ratios calculated
        """
        metricsdf = pd.DataFrame()
        for name in self.metricnames:
            if name in metrics:
                if metricsdf.empty:
                    metricsdf = self.metricnames[name](colname)
                else:
                    adf = self.metricnames[name](colname)
                    metricsdf[self.fullnames[name]] = adf.iloc[:,1]
        metricsdf = metricsdf.set_index('Subgroup')
        for colname in metricsdf.columns:
            metricsdf = metricsdf.rename(columns={colname: (colname + ' ratio')})
        for rowname in metricsdf.index:
            if rowname != privileged:
                metricsdf.loc[rowname,:] = round(metricsdf.loc[rowname,:] / metricsdf.loc[privileged,:], 4)
        metricsdf = metricsdf.drop(privileged, axis=0) #dropping out the privileged group from this df of ratios
        metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
        if printout:
            megalist = []
            for col in metricsdf.columns:
                megalist.append(metricsdf[col].array.tolist())
            megalist = list(map(list, zip(*megalist)))
            demolen = len('Subgroup')
            for item in metricsdf['Subgroup']: #setting length of 1st column to longest subgroup name
                if len(item) > demolen:
                    demolen = len(item)
            if len(metrics) < 4:
                col_widths = [demolen]
                metricsdf = metricsdf.set_index('Subgroup') #moving Subgroups column back to the index
                for col in metricsdf.columns:
                    col_widths.append(len(col))
                metricsdf.reset_index(inplace=True) #moving Subgroups column from index into its own column
                formatted = ' '.join(['%%%ds |' % (width + 1) for width in col_widths ])[:-1]
                colnames = metricsdf.columns
                #printing out table of ratios
                print(formatted % tuple(colnames))
                for row in megalist:
                    print(formatted % tuple(row))
            else: #when there are alot of metrics in the outputted table
                colwidth = 0
                for col in metricsdf.columns:
                    if len(col) > colwidth:
                        colwidth = len(col) + 1
                formatted = ' %%%ds | %%%ds' % (demolen, colwidth)   
                for i in range(len(metrics)): #cycling through all the metrics
                    print(formatted % ('Subgroup', metricsdf.columns[i+1]))
                    for row in megalist:
                        arow = (row[0], row[i+1])
                        print(formatted % arow)
                    print('\n')
        return metricsdf
    
    def EqualOpp(self, columnlist=[], printout=False):
        """
        Calculate Equal Opportunity metric aka True Positive Rate

        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        tprs = {}
        #if columns are not specified then show metric for all columns
        if not columnlist: 
            columnlist = self.democols
        #if there was only one input and the columnlist isn't a list, then make it a list
        elif not isinstance(columnlist, list): 
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fncount[key]) == 0:
                        tprs[key] = 0
                    else:
                        tprs[key] = round((self.tpcount[key] / (self.tpcount[key] + self.fncount[key])), 4)
        if printout: #print out a table if the user specifies to do so
            self._printtable(self.fullnames['TPR'], tprs)
        return self._todf(tprs, 'TPR') #export output as a dataframe
    
    def PredEqual(self, columnlist=[], printout=False):
        """
        Calculate Predictive Equality metric aka False Positive Rate

        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        fprs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fpcount[key] + self.tncount[key]) == 0:
                        fprs[key] = 0
                    else:
                        fprs[key] = round(self.fpcount[key] / (self.fpcount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FPR'], fprs)
        return self._todf(fprs, 'FPR')
    
    def StatParity(self, columnlist=[], printout=False):
        """
        Calculate Statistical Parity metric aka predicted value per subgroup
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        sparity = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    sparity[key] = round((self.tpcount[key] + self.fpcount[key]) / 
                                         (self.tpcount[key] + self.fpcount[key] + 
                                          self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['STP'], sparity)
        return self._todf(sparity, 'STP')
    
    def PredParity(self, columnlist=[], printout=False):
        """
        Calculate Predictive Parity metric aka Positive Predictive Value
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        pparity = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fpcount[key]) == 0:
                        pparity[key] = 0
                    else:
                        pparity[key] = round(self.tpcount[key] / (self.tpcount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['PPV'], pparity)
        return self._todf(pparity, 'PPV')

    def Accuracy(self, columnlist=[], printout=False):
        """
        Calculate Accuracy metric
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        acc = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    acc[key] = round((self.tpcount[key] + self.tncount[key]) / 
                                     (self.tpcount[key] + self.fpcount[key] + 
                                      self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['ACC'], acc)
        return self._todf(acc, 'ACC')

    def TrueNeg(self, columnlist=[], printout=False):
        """
        Calculate True Negative Rate
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        tnrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tncount[key] + self.fpcount[key]) == 0:
                        tnrs[key] = 0
                    else:
                        tnrs[key] = round(self.tncount[key] / (self.tncount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['TNR'], tnrs)
        return self._todf(tnrs, 'TNR')

    def NegPV(self, columnlist=[], printout=False):
        """
        Calculate Negative Predictive Value
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        npv = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tncount[key] + self.fncount[key]) == 0:
                        npv[key] = 0
                    else:
                        npv[key] = round(self.tncount[key] / (self.tncount[key] + self.fncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['NPV'], npv)
        return self._todf(npv, 'NPV')

    def FalseNeg(self, columnlist=[], printout=False):
        """
        Calculate False Negative Rate
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        fnrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fncount[key] + self.tpcount[key]) == 0:
                        fnrs[key] = 0
                    else:
                        fnrs[key] = round(self.fncount[key] / (self.fncount[key] + self.tpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FNR'], fnrs)
        return self._todf(fnrs, 'FNR')

    def FalseDis(self, columnlist=[], printout=False):
        """
        Calculate False Discovery Rate
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        fdrs = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fpcount[key] + self.tpcount[key]) == 0:
                        fdrs[key] = 0
                    else:
                        fdrs[key] = round(self.fpcount[key] / (self.fpcount[key] + self.tpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FDR'], fdrs)
        return self._todf(fdrs, 'FDR')

    def FalseOm(self, columnlist=[], printout=False):
        """
        Calculate False Omission Rate
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        fors = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.fncount[key] + self.tncount[key]) == 0:
                        fors[key] = 0
                    else:
                        fors[key] = round(self.fncount[key] / (self.fncount[key] + self.tncount[key]), 4)
        if printout:
            self._printtable(self.fullnames['FOR'], fors)
        return self._todf(fors, 'FOR')

    def ThreatScore(self, columnlist=[], printout=False):
        """
        Calculate Threat Score
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        threats = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    if (self.tpcount[key] + self.fncount[key] + self.fpcount[key]) == 0:
                        threats[key] = 0
                    else:
                        threats[key] = round(self.tpcount[key] /
                                             (self.tpcount[key] + self.fncount[key] + self.fpcount[key]), 4)
        if printout:
            self._printtable(self.fullnames['TS'], threats)
        return self._todf(threats, 'TS')

    def FScore(self, columnlist=[], printout=False):
        """
        Calculate F1 Score
        
        Inputs
        columnlist <class 'list'>: list of strings of the names of the demographic columns to calculate the metric for
        printout <class 'bool'>: boolean of whether or not to print out a table of the metric

        Outputs
        adf <class 'pandas.core.frame.DataFrame'>: table of the metric calculated for each demographic subgroup
        """
        fscores = {}
        if not columnlist:
            columnlist = self.democols
        elif not isinstance(columnlist, list):
            columnlist = [columnlist]
        for col in self.df:
            if col in columnlist:
                adf = self.df[col]
                demokeys = adf.unique()
                for key in demokeys:
                    ppv = self.PredParity()
                    pvalue = ppv.loc[ppv['Subgroup'] == key, self.fullnames['PPV']].iloc[0]
                    tpr = self.EqualOpp()
                    tvalue = tpr.loc[tpr['Subgroup'] == key, self.fullnames['TPR']].iloc[0]
                    if (pvalue + tvalue) == 0:
                        fscores[key] = 0
                    else:
                        fscores[key] = round(2 * pvalue * tvalue / (pvalue + tvalue), 4)
        if printout:
            self._printtable(self.fullnames['FS'], fscores)
        return self._todf(fscores, 'FS')

    def ReadTruths(self):
        """
        Output dataframe table with more human readable truth column
        """
        df2 = self.df.copy()
        df2.loc[df2['truths'] == 3, 'truths'] = 'True Positive'
        df2.loc[df2['truths'] == 2, 'truths'] = 'False Positive'
        df2.loc[df2['truths'] == 1, 'truths'] = 'False Negative'
        df2.loc[df2['truths'] == 0, 'truths'] = 'True Negative'
        return df2

    

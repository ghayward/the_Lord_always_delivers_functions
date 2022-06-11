"""

These are some functions I find useful. 

Thanks you to God, the coders, and also my Mom.

Errors, they be mine. 

That's a Fried Chicken Man. 

"""

"""

👇🏾 Function 1

- This first function is called hayward_percentiler_Python()
- It helps me get a quick visual on how outliers are exploding the data.

"""

def hayward_percentiler_Python(dataframe, feature_ie_column_to_check):
    percentile_check = [1,25,50,75,90,95,97.0,98.0,99.0,99.7,99.90, 99.99]
    percentile_values = []
    y_pos = np.arange(len(percentile_check))
    for i in percentile_check:
        percentile = i
        value = round(np.percentile(dataframe[feature_ie_column_to_check], i),2)
        percentile_values.append(value)
        print('Percentile '+str(i)+": "+str(value))
    fig, axs = plt.subplots(ncols=2, figsize=(14,4))
    #using a subplot method coupled with an inline parameter to have high resolution
    axs[1].set_axis_off()
    axs[0].bar(y_pos, percentile_values, align='center', alpha=1)
    axs[0].set_ylabel('Values')
    axs[0].set_xticks(y_pos)
    axs[0].set_xticklabels(percentile_check, fontsize=10)
    axs[0].set_xlabel('Percentile')
    plt.suptitle('Value By Percentile: Understanding Outliers',x=0.44, \
                 horizontalalignment='right' )
    axs[0].set_title(feature_ie_column_to_check, fontweight = 'bold')
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.show()

"""

👇🏾 Function 2

- This second function is called hayward_distribution_visualizer_and_fitter()
- It helps me get a quick look at the distribution, how normal the distribution is, and what kinds of more generalizable distributions best describe this data.

"""

def hayward_distribution_visualizer_and_fitter(df, col_str, bins):
    #this part makes the histogram:
    fig, axs = plt.subplots(figsize=(12,4))
    plt.hist(df[col_str], bins=bins)
    plt.title('Histogram of {}'.format(col_str), fontweight = 'bold')
    plt.ylabel('Frequency')
    plt.xlabel('Values')
    plt.show()
    #this part fits the distribution to the histogram:
    f = Fitter(df[col_str], distributions=['gamma', 'beta', 'rayleigh', 'norm', 'pareto', 'uniform', \
                         'logistic', 'expon', 'chi2', 'pearson3'], timeout = 30) #verbose = False
    f.fit()
    f.summary()
    #this part makes the probability plot
    fig, axs = plt.subplots(ncols=2, figsize=(12,4))
    axs[1].set_axis_off()
    stats.probplot(df[col_str], plot=axs[0])
    plt.show()


"""

👇🏾 Function 3

- This third function is called df_iqr_adjust_romain_hayward()
- This critical function is what I use to eliminate outliers in a dataframe.
- The definition for what constitutes an outlier was first laid out in the following book:
- "Statistics in a Nutshell, 2nd Edition" by Sarah Boslaugh
---- Link: https://www.oreilly.com/library/view/statistics-in-a/9781449361129/
- I wrote this function after reading a post by a SWE called Romain here.
---- Link: https://www.back2code.me/2017/08/outliers/

"""

def df_iqr_adjust_romain_hayward(df, list_of_columns_to_modify):
    iqr_df = df
    for i in list_of_columns_to_modify:
        Q1 = iqr_df[i].quantile(0.25)
        Q3 = iqr_df[i].quantile(0.75)
        IQR = Q3 - Q1
        iqr_df = iqr_df.query('(@Q1 - 1.5 * @IQR) <= %s <= (@Q3 + 1.5 * @IQR)' % i)
    return iqr_df


"""

👇🏾 Function 4

- This fourth function is called celery_hayward_spliced_percentiler()
- It was written by Étienne Célèry in response to a question I asked on Stack Overflow.
---- Link: https://stackoverflow.com/questions/71608038/how-to-re-write-lambda-function-via-pandas-apply-method-to-beat-famous-val/71608269#71608269
- Below, I've particularized the function for this exam, and it's very useful in computing what I call "sub-percents" in multi-level indexed Pandas dataframes. This function will be a useful component of my forthcoming mega functions.

"""

def celery_hayward_spliced_percentiler(assignment_value,count_value):
     if assignment_value == 'treatment':
          return round((100.0*count_value) / (size_count_treatment) ,2)
     else:
          return round((100.0*count_value) / (size_count_control) ,2)

"""

👇🏾 Function 5

- This fifth function is called get_df_name()
- I found this function on Stack Overflow here. It was written by 'Min'.
---- Link: https://stackoverflow.com/questions/31727333/get-the-name-of-a-pandas-dataframe/50620134#50620134
- As you will see, I'll have many dataframes flying around, and this function gets helps resolve a pain in Python and Pandas, which is that it's hard to convert an object or a variable's name into a string, so it's hard to stay organized.

"""

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name


"""

👇🏾 Function 6

- This sixth function is called hayward_two_sample_proportion_data_uploader()
- This function will:
---- first sanity the counts of the data for imbalances, and print the results,
---- then it will perform a two sample proportion test in R,
---- finally, it will upload the key results to a series of empty lists,
---- which I will then covert later into a Pandas dataframe.
-------- This dataframe will look like a report that we can quickly read to discuss results.
- Critically, as discussed above, to avoid Simspon's Paradox, this function calculates everything on a per guest country basis.
- This function is best utilized in a for loop.

"""

def hayward_two_sample_proportion_data_uploader(df, country_letter_string, binary_success_column_string):
    #take a filtered view of the df
    df_temp = df[df.guest_country == country_letter_string]
     
    print("Country: "+str(country_letter_string))  
    print("")
    
    #let's take a quick count of the arms of the test
    size_check = df_temp.groupby("assignment").agg(total_guest_count=pd.NamedAgg(\
                                                                column='guest_id',
                                                                aggfunc='count'))
    size_count = size_check.total_guest_count.sum()
    size_count_control = size_check.loc['control', 'total_guest_count'].sum()
    size_count_treatment = size_check.loc['treatment', 'total_guest_count'].sum()
    
    size_check['percent'] = size_check.apply(lambda x: round(100.0*(x/ size_count),2) , axis=1)
    
    display(size_check)
    
    def celery_hayward_spliced_percentiler(assignment_value,count_value):
        if assignment_value == 'treatment':
            return round((100.0*count_value) / (size_count_treatment) ,2)
        else:
            return round((100.0*count_value) / (size_count_control) ,2)
    
    #now let's do a sanity check on the PLUF
    pluf_check = df_temp.groupby(["assignment","payment_option"], dropna=False)\
                    .agg(total_guest_count=pd.NamedAgg(\
                                        column='guest_id',
                                        aggfunc='count'))#.style.format('{0:,.0f}')
                #thanks https://stackoverflow.com/a/61922965/11736959 for dropna
    pluf_check['sub_percent'] = pluf_check.apply(lambda x: celery_hayward_spliced_percentiler(x.index,\
                                                                                x.total_guest_count) , axis=1)
    #theLordalwaysdelivers
    display(pluf_check)
    
    #now let's split up the proportions
    
    treatment_bookings =\
    df_temp[df_temp.assignment == \
                           'treatment'][binary_success_column_string].sum()
    treatment_impressions =\
    df_temp[df_temp.assignment == \
                            'treatment'][binary_success_column_string].count()
    control_bookings =\
    df_temp[df_temp.assignment == \
                               'control'][binary_success_column_string].sum()
    control_impressions =\
    df_temp[df_temp.assignment == \
                                'control'][binary_success_column_string].count()
    
    #now let's run R
    
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter): 
        r.assign("treatment_bookings", treatment_bookings) 
        r.assign("treatment_impressions", treatment_impressions)
        r.assign("control_bookings", control_bookings)
        r.assign("control_impressions", control_impressions)
        hello_world_R_2samp_prop = r("""
          dilla <- prop.test(
                  x = c(control_bookings, treatment_bookings), 
                  n = c(control_impressions, treatment_impressions), 
                  p = NULL,
                  alternative = "two.sided",
                  conf.level = 0.95, correct = TRUE)
          dilla        
          """) 
        p_value_readout_prop = r('round(dilla$p.value,5)')
        conf_int_prop = r('round(dilla$conf.int,4)')
        parameter_df_prop = r('round(dilla$parameter,4)')
        group_estimates_prop = r('round(dilla$estimate,4)')
    print(hello_world_R_2samp_prop)
        
    #now let's upload the payload
    guest_country_prop.append(country_letter_string)
    def get_df_name(a_dataframe_yes): #thanks amazing -> https://stackoverflow.com/a/50620134/11736959
        name =[x for x in globals() if globals()[x] is a_dataframe_yes][0]
        return name
    data_name_prop.append(get_df_name(df))
    success_variable.append(binary_success_column_string)
    control_prop.append(group_estimates_prop[0])
    treatment_prop.append(group_estimates_prop[1])
    degrees_of_freedom_prop.append(parameter_df_prop[0])
    conf_int_lower_prop.append(conf_int_prop[0])
    conf_int_upper_prop.append(conf_int_prop[1])
    p_value_prop.append(p_value_readout_prop[0])
    
"""

👇🏾 Function 7

- This seventh function is called hayward_diff_of_means_data_party_uploader()
- This function will:
---- first sanity the counts of the data for imbalances, and print the results,
---- then it will perform a Welch two sample t-test test in R,
---- finally, it will upload the key results to a series of empty lists,
---- which I will then covert later into a Pandas dataframe.
-------- This dataframe will look like a report that we can quickly read to discuss results.
- Critically, as discussed above, to avoid Simspon's Paradox, this function calculates everything on a per guest country basis.
- This function is best utilized in a for loop.
- This function is basically the same as the last one, but it just uses a different statistical test.

"""

def hayward_diff_of_means_data_party_uploader(df, country_letter_string, column_to_analyze):
    #take a filtered view of the df
    df_temp = df[df.guest_country == country_letter_string]
     
    print("Country: "+str(country_letter_string))  
    print("")
    
    #let's take a quick count of the arms of the test
    size_check = df_temp.groupby("assignment").agg(total_guest_count=pd.NamedAgg(\
                                                                column='guest_id',
                                                                aggfunc='count'))
    size_count = size_check.total_guest_count.sum()
    size_count_control = size_check.loc['control', 'total_guest_count'].sum()
    size_count_treatment = size_check.loc['treatment', 'total_guest_count'].sum()
    
    size_check['percent'] = size_check.apply(lambda x: round(100.0*(x/ size_count),2) , axis=1)
    
    display(size_check)
    
    def celery_hayward_spliced_percentiler(assignment_value,count_value):
        if assignment_value == 'treatment':
            return round((100.0*count_value) / (size_count_treatment) ,2)
        else:
            return round((100.0*count_value) / (size_count_control) ,2)
    
    #now let's do a sanity check on the PLUF
    pluf_check = df_temp.groupby(["assignment","payment_option"], dropna=False)\
                    .agg(total_guest_count=pd.NamedAgg(\
                                        column='guest_id',
                                        aggfunc='count'))#.style.format('{0:,.0f}')
                #thanks https://stackoverflow.com/a/61922965/11736959 for dropna
    pluf_check['sub_percent'] = pluf_check.apply(lambda x: celery_hayward_spliced_percentiler(x.index,\
                                                                                x.total_guest_count) , axis=1)
    #theLordalwaysdelivers
    display(pluf_check)
    
   
    #now let's run R
    
    stats_R = importr("stats") #thanks https://stackoverflow.com/a/9614183/11736959
    base_R = importr("base") 
    with localconverter(ro.default_converter + pandas2ri.converter): 
        r.assign("df_temp_R", df_temp) 
        hello_world_R = r(f"""
        Lord_Delivers = t.test({column_to_analyze} ~ assignment, data = df_temp_R )
        Lord_Delivers
        """)
        p_value_readout = r("round(Lord_Delivers$p.value,5)")
        conf_int = r("round(Lord_Delivers$conf.int,2)")
        parameter_df = r("round(Lord_Delivers$parameter,0)")
        group_estimates = r("round(Lord_Delivers$estimate,2)")
    print(hello_world_R)
    
    
    
    #now we upload the payload
    guest_country.append(country_letter_string)
    def get_df_name(a_dataframe_yes): #thanks amazing -> https://stackoverflow.com/a/50620134/11736959
        name =[x for x in globals() if globals()[x] is a_dataframe_yes][0]
        return name
    data_name.append(get_df_name(df))
    feature.append(column_to_analyze)
    control_mean.append(group_estimates[0])
    treatment_mean.append(group_estimates[1])
    degrees_of_freedom.append(parameter_df[0])
    conf_int_lower.append(conf_int[0])
    conf_int_upper.append(conf_int[1])
    p_value.append(p_value_readout[0])

"""

👇🏾 Function 8

- This eighth function is called hayward_experiment_duration_checker()
- It'll come in handy for picking the duration of an experiment.

"""

def hayward_experiment_duration_checker(dollar_size_effect,df, column_to_analyze_string, country_string):
    d = (dollar_size_effect) / (df[df.guest_country == country_string][column_to_analyze_string].std())
    
    stats_R = importr("stats") 
    base_R = importr("base")
    pwr_R = importr("pwr", lib_loc="/Library/Frameworks/R.framework/Versions/4.1/Resources/library")
    with localconverter(ro.default_converter + pandas2ri.converter): 
      r.assign("d_R", d) 
      hello_world_R = r("""
          the_bus = pwr.t.test(d = d_R, power = 0.80, sig.level=0.05)
      """) 
    n_people = r("the_bus$n")
    n_people_python = n_people[0] * 2 #because the n is in each group, so the full experiment doubles it
    
    total_peeps_in_country = df[df.guest_country == country_string].guest_id.nunique()
    days_duration_of_total_peeps_count = df[df.guest_country == country_string].date_of_impression.nunique()
    
    new_people_per_day = round(total_peeps_in_country / days_duration_of_total_peeps_count,0)
    
    days_needed = round(n_people_python /new_people_per_day ,0)
    
    return days_needed



"""

#TheLordAlwaysDelivers!

:)

God Bless You // Sending the Good Vibes!
"""

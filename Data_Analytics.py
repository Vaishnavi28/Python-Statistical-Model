
# coding: utf-8

# In[54]:
#To Calculate the mean:
def mean(string_input):
    is_valid_input = False
    while not is_valid_input:
        try:
            input_list = string_input.split() #splits the input string on spaces
            # process string elements in the list and make them integers
            input_list = [float(a) for a in input_list]
            print input_list;
            is_valid_input = True
            return float(sum(input_list))/len(input_list)
        except:
            print 'Invalid Entry!,enter proper numbers'
            string_input = raw_input("Enter numbers delimited by space as string,example \"1 2 3\"")
            is_valid_input = False

#To Calculate the mean:
def mean():
    string_input = raw_input('Enter the numbers with spaces to calculate the mean')
    try :
            input_list = string_input.split() #splits the input string on spaces
            # process string elements in the list and make them integers
            input_list = [float(a) for a in input_list]
            print input_list;
            Mean=float(sum(input_list))/len(input_list)
            print ("Mean = %f" % Mean)
    except:
            print 'Invalid Entry!'
            mean()         
            
         

    


# In[58]:

#To calculate Median:
def median():
    lst=raw_input('Enter the numbers with spaces to calculate the median')
    try:
        lst = lst.split()
        lst=[float(a) for a in lst]
        print lst;
        lst = sorted(lst)     
        if len(lst) < 1:
            print 'Invalid Entry!'
            median()
        if len(lst) %2 == 1:
                lst_odd= lst[((len(lst)+1)/2)-1]
                print ("Median= %f" % lst_odd)
        else:
                lst_even= float(sum(lst[(len(lst)/2)-1:(len(lst)/2)+1]))/2.0
                print ("Median= %f" % lst_even)
    except:
        print 'Invalid Entry!'
        median()
            
        
        


# In[60]:

#To Calculate the mode:
def mode():
    mode_input = raw_input('Enter the numbers with spaces to calculate the mode')
    try :
            mode_list = mode_input.split() 
            mode_list = [float(a) for a in mode_list]
            mode_list=sorted(mode_list)
            print mode_list
            Mode= mode_list[len(mode_list)-1]-mode_list[0]
            print ("Mode = %f" % Mode)
    except:
            print 'Invalid Entry!'
            mode()  


# In[71]:

def percentile():
    import numpy as np
    percentilevalues== raw_input('Enter the numbers with spaces to calculate the mode')
    try:
        finalvalues = percentilevalues.split() 
        finalvalues = [float(a) for a in finalvalues]
        finalvalues=sorted(finalvalues)
        print finalvalues
        for i in finalvalues.values():
            print np.percentile(map(int,i),95)
    except:
        print'Invalid Entry!'
        percentile()
    


# In[ ]:




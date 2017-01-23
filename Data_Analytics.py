
# coding: utf-8

# In[71]:

def mean(input_list):
    mean= float(sum(input_list))/len(input_list)
    return mean
    
def median(input_list):
    lst = sorted(input_list)
    if len(lst) %2 == 1:
        lst_odd= lst[((len(lst)+1)/2)-1]
        print ("Median= %.3f" % lst_odd)
    else:
        lst_even= float(sum(lst[(len(lst)/2)-1:(len(lst)/2)+1]))/2.0
        print ("Median= %.3f" % lst_even)

def mode(input_list):#ccheck properly
    from collections import Counter
    data = Counter(input_list)
    data.most_common()  # Returns all unique items and their counts
    mode=data.most_common(1)
    print "Mode of data getting repeated for no.of times is displayed:"
    print mode
    
def percentile(prcntl,p):
    import math
    try:
        p=float(p)
        if p<=0:
            raise ValueError('Please enter p value greater than 0 and lesser than 100')
        elif p>=100:
            raise ValueError('Please enter p value greater than 0 and lesser than 100')
        n=len(prcntl)
        pp=n*(p/100)
        if pp.is_integer():
            pp=int(pp)
            prcntl_value=(prcntl[pp-1]+ prcntl[pp])/2
            return prcntl_value
        else :
            pp=math.ceil(pp)
            pp=int(pp)
            prcntl_value=prcntl[pp-1]
            return prcntl_value
    
    except (TypeError, ValueError) as e:
        print e

        
def quartile(qrt,q): 
    try:
        q=int(q)
        if q==1:
            qrt_1=percentile(qrt,25)
            return qrt_1
        elif q==2:
            qrt_2=percentile(qrt,50)
            return qrt_2
        elif q==3:
            qrt_3=percentile(qrt,75)
            return qrt_3
        else:
            raise ValueError('Invalid entry for quartile: 1,2,3 are only accepted')
    
    except (TypeError, ValueError) as e:
        print e
        

def iqr(input_data):
    Q1=quartile(input_data,1)
    Q3=quartile(input_data,3)
    iqr_final=Q3-Q1
    print ("InterQuartile Range = %.2f" % iqr_final)

def num_range(mode_list):
    mode_list=sorted(mode_list)
    Mode= mode_list[len(mode_list)-1]-mode_list[0]
    print ("Range = %.2f" % Mode)

def var_p(input_list):
    n=len(input_list)
    sum=0
    for i in xrange(0, n):
        sum+=(input_list[i]-mean(input_list))**2.
    var_p=sum/n
    return var_p

def var_s(input_list):
    n=len(input_list)
    sum=0
    for i in xrange(0, n):
        sum+=(input_list[i]-mean(input_list))**2.
    var_s=sum/n-1
    return var_s

from math import sqrt

def std_p(input_list):
    std_p=sqrt(var_p(input_list))
    return std_p

def std_s(input_list):
    std_s=(sqrt(var_s(input_list)))
    return std_s

def skewness(input_list):
    n=(len(input_list))
    x=input_list
    N=(float(n)/((n-1)*(n-2)))
    xbar=mean(input_list)
    std=std_s(input_list)
    sum=0
    for i in xrange(0,n):
        sum+=((x[i]-xbar)/std)**3
    Skew=N*sum
    return Skew

def cv_p(input_list):
    cv_p=(std_p(input_list)/mean(input_list))*100
    return cv_p

def cv_s(input_list):
    cv_s=(std_s(input_list)/mean(input_list))*100
    return cv_s

def z_score(input_list,x):
    z=(x-mean(input_list))/std_p(input_list)
    return z

def outlier_z(input_list):
    Z_data = list()
    Z_data = [z_score(input_list,item) for item in input_list] 
    #print ",".join([str(z) for z in Z_data])
    outlier=list()
    for i in xrange(0,len(input_list)):
        if Z_data[i] < -3 or Z_data[i] > 3:
            outlier.append(Z_data[i])
    print ",".join([str(z) for z in outlier])
    return outlier


def cov_p(x,y):#check cov calculation. validate it
    n=len(x)
    mean_x=mean(x)
    mean_y=mean(y)
    sum = 0
    for i in xrange(0, n):
        sum+= (x[i]-mean_x)*(y[i]-mean_y)
    cov_p=sum/n
    return cov_p


def cov_s(x,y):
    n=len(x)
    mean_x=mean(x)
    mean_y=mean(y)
    sum = 0
    for i in xrange(0, n):
        sum+= (x[i]-mean_x)*(y[i]-mean_y)
    cov_s=sum/(n-1)
    return cov_s

def r_pearson_p(x,y):
    cor_p=cov_p(x,y)/(std_p(x)*std_p(y))
    return cor_p

def r_pearson_s(x,y):
    cor_s=cov_s(x,y)/(std_s(x)*std_s(y))
    return cor_s

def factorial (n):
    try:
        if (n < 0):
            raise ValueError('factorial() not defined for negative values')
        elif type(n) != int:
            raise TypeError('an integer is required')
        elif n==0:
            return 1
        else:                     
            i = 1
            prod = 1
            while (i <= n):
                prod = int(prod * i)
                i = i + 1
            return prod
        
    except (TypeError, ValueError) as e:
        print e
def permutation(N,n):
    if n > N:
        print("PLease enter proper values & call the function again:")
        exit
        
    pt=int(factorial(N)/factorial(N-n))
    return pt

def combination(N,n):
    if n > N:
        print("please enter proper values and call the function again:")
        exit
    for n in xrange (n, N): 
        cm=int(factorial(N)/(factorial(N-n)*factorial(n)))
        return cm
def binomial_dist(n,p,x,iscdf):
    if not iscdf:
        pmf=combination(n,x)* (p**x) *((1-p)**(n-x))
        return pmf
    else:
        y = 1.0 - p
        a = n - x
        b = x + 1
        c = a + b - 1
        c=int(c)
        a=int(a)
        b=int(b)
        cdf = 0.0
        for j in xrange(a, c + 1):
            cdf += combination(c,j)                     * y**j * (1 - y)**(c-j)
        return cdf

def poisson_dist(mu,x,iscdf):
    if not iscdf:
        pmf=((math.e**(-mu)) * ((mu**x)/factorial(x)))
        return pmf
    else:
        sum = 0
        for i in xrange(0, x+1):
            sum += mu**i/factorial(i)

        cdf=sum*math.e**(-mu)
        return cdf

def hypergeometric_dist(N,r,n,x,iscdf):
    if not iscdf:
        #pmf=[ rCx ] [ N-r C n-x ] / [ NCn ]
        pmf= float(combination(r,x) * (combination(N-r,n-x) ))/combination(N,n)
        return pmf 
    else:
        sum = 0
        for i in xrange(0, x+1):
            sum += float(combination(r,i) * (combination(N-r,n-i)))

        cdf=sum/combination(N,n)
        return cdf
       
def uniform_dist(a,b,x,iscdf):
    if a>b:
        print("a>b, Enter proper values for lower and upper limits")
    if not iscdf:
        if x<a or x>b:
            pdf=0
            return pdf
        else:
            pdf=1./(b-a)
            return pdf
    else:
        if x<a:
            cdf=0
            return cdf
        elif x>b:
            cdf=1
            return cdf
        else:
            cdf=float(x-a)/(b-a)
            return cdf

import math
def exponential_dist(mu,x,iscdf):
    if not iscdf:
        pdf=mu*(math.e**(-mu*x))
        return pdf
    else:
        cdf=1- (math.e**(-mu*x))
        return cdf
    
def test_clt_with_uniform_dist(n,t):
    import numpy as np
    import matplotlib.pyplot as plt
    sample=np.random.uniform(low=0.0, high=1.0, size=(t,n))
    sample_mean = np.mean(sample, axis=1)
    grand_mean = np.mean(sample_mean)
    sigma = np.std(sample_mean, ddof=1)
    get_ipython().magic(u'matplotlib inline')
    print"Grand Mean: %.3f"% grand_mean
    print "Standard Deviation : %.3f"% sigma
    plt.hist(sample_mean,bins=100,normed=1)
    plt.xlabel('Sample Mean')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ CLT:}\ \mu=%.3f,\ \sigma=%.3f$' %(grand_mean, sigma))
    plt.grid(True)
    plt.show()
    

def ci_mean(input_list,a,isPopDist):
    if not isinstance (input_list,list):
        print "The first parameter needs to be a list"
        raise ValueError("The first parameter needs to be a list")
    if not isinstance (isPopDist,bool):
        print "The third parameter needs to be a bool"
        raise ValueError("The third parameter needs to be a bool")
          
    from scipy.stats import t
    from scipy.stats import norm
    #from numpy import average, std
    from math import sqrt
    avg=mean(input_list)
    alpha=1-a/2.
    if not isPopDist:
        data = input_list
        stddev = std_s(data)
        n=len(data)
        critval=t.ppf(alpha,(n-1))
        MOE=critval * (stddev / sqrt(n-1))
        print("T-critical value:")              # Check the t-critical value
        print(critval)
        # sum mean to the confidence interval
        ci = [avg - MOE ,avg + MOE]
        print "Mean: %.2f" % avg
        print "Lower & Upper bounds of Confidence Interval: %f, %f" % (ci[0], ci[1])
    else:
        data=input_list
        z_critical = norm.ppf(alpha)  # Get the z-critical value*
        print("z-critical value:")              # Check the z-critical value
        print(z_critical)                        
        pop_stdev = std_p(data)  # Get the population standard deviation
        MOE = z_critical * (pop_stdev/sqrt(len(data)))
        confidence_interval = (avg - MOE,avg + MOE) 
        print ("Confidence Interval:")
        print(confidence_interval)
        
def ci_proportion(pbar,n,a):
          
    from scipy.stats import t
    from scipy.stats import norm
    #from numpy import average, std
    from math import sqrt
    alpha=1-a/2.
    z_critical = norm.ppf(alpha)  # Get the z-critical value*
    print("z-critical value:")              # Check the z-critical value
    print(z_critical)                        
    MOE = z_critical * sqrt((pbar*(1-pbar)) / (n))
    confidence_interval = ((pbar - MOE)*100,(pbar + MOE)*100) 
    if confidence_interval[0]<0:
        print "Lower bound = 0"
        print "Upper bound" %(confidence_interval[1])
    else:        
        print ("Confidence Interval:")
        print(confidence_interval)

def hypo_test_for_mean(xbar,mu,n,a,SD,isPopSD,tail):
    from scipy.stats import t
    from scipy.stats import norm
    from math import sqrt
    if not isPopSD:
        SE=SD/sqrt(n)
        t_value=(xbar-mu)/SE
        print ("T-Value = %.3f" % t_value)
        if tail==-1:
            print"Lower Tail test"
            p_value=t.cdf(t_value,n-1)
            print ("P-value= %.3f" % p_value)
            critical_t=t.ppf(a,(n-1))
            print ("Critical-T= %.3f" % critical_t)
            if p_value <= a and t_value < critical_t:
                print ("False:Reject H0")
            else:
                print "True:Do NOT Reject H0"
        if tail== 1:
            print"Upper Tail test"
            p_value=1-t.cdf(t_value,n-1)
            print ("P-value= %.3f" % p_value)
            critical_t=t.ppf((1-a),(n-1))
            print ("Critical-T= %.3f" % critical_t)
            if p_value <= a and t_value >= critical_t:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail==0:
            print"Two Tailed test"
            p_value=2*(t.cdf(t_value,n-1))
            print ("P-value= %.3f" % p_value)
            Upper_critical_t=t.ppf((1-a/2),(n-1))
            Lower_critical_t=t.ppf((a/2),(n-1))
            print ("Upper Critical-T= %.3f" % Upper_critical_t)
            print ("Lower Critical-T= %.3f" % Lower_critical_t)
            if p_value <= a :
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
    else:
        SE=SD/sqrt(n)
        z_value=(xbar-mu)/SE
        print ("Z-Value = %.3f" % z_value)
        if tail==-1:
            print"Lower Tail test"
            p_value=norm.cdf(z_value)
            print ("P-value= %.3f" % p_value)
            critical_z=norm.ppf(a)
            print ("Critical-Z= %.3f" % critical_z)
            if p_value <= a and z_value <= critical_z:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail== 1:
            print"Upper Tail test"
            p_value=1-norm.cdf(z_value)
            print ("P-value= %.3f" % p_value)
            critical_z=norm.ppf(1-a)
            print ("Critical-Z= %.3f" % critical_z)
            if p_value <= a and z_value >= critical_z:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail==0:
            print"Two Tailed test"
            p_value=2*(norm.cdf(z_value))
            print ("P-value= %.3f" % p_value)
            Upper_critical_z=norm.ppf((1-a/2))
            Lower_critical_z=norm.ppf(a/2)
            print ("Upper Critical-Z= %.3f" % Upper_critical_z)
            print ("Lower Critical-Z= %.3f" % Lower_critical_z)
            if p_value <= a :
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
 
            
def hypo_test_for_proportion(p,p0,n,a,tail):
    from scipy.stats import t
    from scipy.stats import norm
    from math import sqrt
    SE=sqrt((p*(1-p))/n)
    z_value=(p-p0)/SE
    print ("Z-value= %.3f" % z_value)
    if tail==-1:
        p_value=norm.cdf(z_value)
        print ("P-value= %.3f" % p_value)
        critical_z=norm.ppf(a)
        print ("Critical-Z= %.3f" % critical_z)
        if p_value <= a and z_value <= critical_z:
            print "False:Reject H0"
        else:
            print "True:Do NOT Reject H0"
    if tail== 1:
        print"Upper Tail test"
        p_value=1-norm.cdf(z_value)
        print ("P-value= %.3f" % p_value)
        critical_z=norm.ppf(1-a)
        print ("Critical-Z= %.3f" % critical_z)
        if p_value <= a and z_value >= critical_z:
            print "False:Reject H0"
        else:
            print "True:Do NOT Reject H0"
    if tail==0:
        print"Two Tailed test"
        p_value=2*(norm.cdf(z_value))
        print ("P-value= %.3f" % p_value)
        Upper_critical_z=norm.ppf((1-a/2))
        Lower_critical_z=norm.ppf(a/2)
        print ("Upper Critical-Z= %.3f" % Upper_critical_z)
        print ("Lower Critical-Z= %.3f" % Lower_critical_z)
        if p_value <= a :
            print "False:Reject H0"
        else:
            print "True:Do NOT Reject H0"

def power_in_hypo_test_for_mean(xbar,mu,n,a,SD,isPopSD,tail):
    from scipy.stats import t
    from scipy.stats import norm
    from math import sqrt
    if not isPopSD:
        SE=SD/sqrt(n)
        t_value=(xbar-mu)/SE
        print ("T-Value = %.3f" % t_value)
        if tail==-1:
            print"Lower Tail test"
            Beta=t.cdf(t_value,n-1)
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )
        if tail== 1:
            print"Upper Tail test"
            Beta=1-t.cdf(t_value,n-1)
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )
        if tail==0:
            print"Two Tailed test"
            p_value=2*(t.cdf(t_value,n-1))
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )
    else:
        SE=SD/sqrt(n)
        z_value=(xbar-mu)/SE
        print ("Z-Value = %.3f" % z_value)
        if tail==-1:
            print"Lower Tail test"
            Beta=norm.cdf(z_value)
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )
        if tail== 1:
            print"Upper Tail test"
            Beta=1-norm.cdf(z_value)
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )
        if tail==0:
            print"Two Tailed test"
            Beta=2*(norm.cdf(z_value))
            power=1-Beta
            print("Power of the hypothesis test is %.3f " % power )

def ci_for_mean_difference((mu1,mu2),(n1,n2),(SD1,SD2),a,isPopDist): 
    from scipy.stats import t
    from scipy.stats import norm
    from math import sqrt
    avg=mu1-mu2
    alpha=1-a/2.
    stddev = sqrt(((SD1*SD1)/n1)+((SD2*SD2)/n2))
    E=mu1-mu2
    if not isPopDist:
        s=(((n1-1)*(SD1*SD1))+((n2-1)*(SD2*SD2)))/(n1+n2-2)
        s_final=sqrt(s*((1./n1)+(1./n2)))
        critval=t.ppf(alpha,(n1+n2-2))
        MOE=critval*s_final     
        ci = [E - MOE ,E + MOE]
        print ("Confidence Interval:")
        print(ci)
    else:
        z_critical = norm.ppf(alpha)  # Get the z-critical value*
        MOE = z_critical * stddev
        confidence_interval = (E - MOE,E + MOE) 
        print ("Confidence Interval:")
        print(confidence_interval)
        
def hypo_test_for_mean_difference((x1,x2),(n1,n2),(SD1,SD2),a,isPopSD,Mudiff,tail):
    from scipy.stats import t
    from scipy.stats import norm
    from math import sqrt
    if not isPopSD:
        s=(((n1-1)*(SD1*SD1))+((n2-1)*(SD2*SD2)))/(n1+n2-2)
        s_final=sqrt(s*((1./n1)+(1./n2)))
        t_value=((x1-x2)-(Mudiff))/s_final
        print ("T-Value = %.3f" % t_value)
        if tail==-1:
            print"Lower Tail test"
            p_value=t.cdf(t_value,n1+n2-2)
            print ("P-value= %.3f" % p_value)
            critical_t=t.ppf(a,(n1+n2-2))
            print ("Critical-T= %.3f" % critical_t)
            if p_value <= a :
                print ("False:Reject H0")
            else:
                print "True:Do NOT Reject H0"
        if tail== 1:
            print"Upper Tail test"
            p_value=1-t.cdf(t_value,n1+n2-2)
            print ("P-value= %.3f" % p_value)
            critical_t=t.ppf((1-a),(n1+n2-2))
            print ("Critical-T= %.3f" % critical_t)
            if p_value <= a:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail==0:
            print"Two Tailed test"
            p_value=2*(t.cdf(t_value,n1+n2-2))
            print ("P-value= %.3f" % p_value)
            Upper_critical_t=t.ppf((1-a/2),(n1+n2-2))
            Lower_critical_t=t.ppf((a/2),(n1+n2-2))
            print ("Upper Critical-T= %.3f" % Upper_critical_t)
            print ("Lower Critical-T= %.3f" % Lower_critical_t)
            if p_value <= a :
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
    else:
        SE=sqrt(((SD1*SD1)/n1) +((SD2*SD2)/n2))
        z_value=((x1-x2)-(Mudiff))/SE
        print ("Z-Value = %.3f" % z_value)
        if tail==-1:
            print"Lower Tail test"
            p_value=norm.cdf(z_value)
            print ("P-value= %.3f" % p_value)
            critical_z=norm.ppf(a)
            print ("Critical-Z= %.3f" % critical_z)
            if p_value <= a:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail== 1:
            print"Upper Tail test"
            p_value=1-norm.cdf(z_value)
            print ("P-value= %.3f" % p_value)
            critical_z=norm.ppf(1-a)
            print ("Critical-Z= %.3f" % critical_z)
            if p_value <= a:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"
        if tail==0:
            print"Two Tailed test"
            p_value=2*(1-norm.cdf(z_value))
            print ("P-value= %.3f" % p_value)
            Upper_critical_z=norm.ppf((1-a/2))
            Lower_critical_z=norm.ppf(a/2)
            print ("Upper Critical-Z= %.3f" % Upper_critical_z)
            print ("Lower Critical-Z= %.3f" % Lower_critical_z)
            if p_value<=a:
                print "False:Reject H0"
            else:
                print "True:Do NOT Reject H0"

def ci_for_proportion_difference((p1,p2),(n1,n2),a):
    from scipy.stats import norm
    from math import sqrt
    E=p1-p2
    SD=sqrt(((p1*(1.-p1))/n1)+((p2*(1.-p2))/n2))
    z=norm.ppf((1-a/2))
    MOE=SD*z
    CI=(E-MOE,E+MOE)
    print "Lower & Upper bounds of Confidence Interval:"
    print CI 
    
def hypo_test_for_proportion_difference((p1,p2),(n1,n2),a,tail):
    from scipy.stats import norm
    from math import sqrt
    z_num=(p1-p2)
    pbar=((n1*p1)+(n2*p2))/(n1+n2)
    n=((1./n1)+(1./n2))
    z_den=sqrt((pbar*(1.-pbar))*n)
    Z_value=z_num/z_den
    print "Z-Value= %.3f" % Z_value
    if tail==-1:
        print"Lower tail test"
        p=norm.cdf(Z_value)
        print  "P-Value= %.3f"%p
        critical_z=norm.ppf(a)
        print "Critical Z= %.3f" %critical_z
        if p<=a:
            print"False: H0 is rejected"
        else:
            print"True: H0 is not rejected"
            
    elif tail==1:
        print"Upper tail test"
        p=1-norm.cdf(Z_value)
        print  "P-Value= %.3f"%p
        critical_z=norm.ppf(1-a)
        print "Critical Z= %.3f" %critical_z
        if p<=a:
            print"False: H0 is rejected"
        else:
            print"True: H0 is not rejected"
    elif tail==0:
        print"Two Tailed test"
        p_value=2*(1-norm.cdf(Z_value))
        print ("P-value= %.3f" % p_value)
        Upper_critical_z=norm.ppf((1-a/2))
        Lower_critical_z=norm.ppf(a/2)
        print ("Upper Critical-Z= %.3f" % Upper_critical_z)
        print ("Lower Critical-Z= %.3f" % Lower_critical_z)
        if p_value<=a:
            print "False:Reject H0"
        else:
            print "True:Do NOT Reject H0"


def ci_for_population_var(s2,n,a):
    from scipy.stats import chi2
    alpha=a/2
    Den_High=chi2.ppf(alpha,(n-1))
    Num=((n-1)*(s2))
    CI_High=Num/Den_High
    Den_Low=chi2.ppf((1-alpha),(n-1))
    CI_Low=Num/Den_Low
    print "Lower & Upper bounds of Confidence Interval: %f, %f" % (CI_Low, CI_High)

def hypo_test_for_population_var(s2,sigma2,n,a,tail):
    from scipy.stats import chi2
    chi=((n-1)*s2)/float(sigma2)
    print "Chi-Square= %.3f" % chi
    p_value = chi2.cdf(chi, n-1)
    if tail==-1:
        print "Lower Tail Test"
        Crit_Chi=chi2.ppf(a,(n-1))
        print "Critical Chi2 Value = %.3f" % Crit_Chi
        print "P-value= %.3f"%p_value
        if p_value<=a:
            print "False: H0 is Rejected"
        else:
            print "True: H0 is NOT Rejected"
    elif tail==1:
        print "Upper Tail Test"
        Crit_Chi=chi2.ppf((1-a),(n-1))
        p_value=1-p_value
        print "Critical Chi2 Value = %.3f" % Crit_Chi
        if p_value<=a:
            print "False: H0 is Rejected"
        else:
            print "True: H0 is NOT Rejected"
       
        
def hypo_test_for_two_population_var((s1,s2),(n1,n2),a,tail):
    from scipy.stats import f
    F= float(s1)/s2
    print "F=%.3f" %F
    p_value = f.cdf(F,n1-1,n2-1)
        
    if tail == -1 :
        print "P-Value= %.3f" % p_value
        F_critical= 1/(f.ppf(1-a,n1-1,n2-1))
        print "F-Critical=%.3f" % F_critical
        if p_value <= a:
            print"False: H0 is rejected"
        else:
            print"True: H0 is not rejected"
    elif tail == 1:
        F_critical = f.ppf(a,n1-1,n2-1)
        print "F-Critical=%.3f" % F_critical
        p_value = 1-p_value
        print "P-Value= %.3f" % p_value
        if p_value <= a:
            print"False: H0 is rejected"
        else:
            print"True: H0 is not rejected"


        


# In[ ]:




import numpy as np
import mesa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class Agent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        a = 5 # shape
        rng = np.random.default_rng()
        self.wealth = 1 - rng.power(a, 1)
        self.job =  rng.binomial(1, 0.25 + 0.5*self.wealth,1)
        self.fraud = rng.binomial(1,0.5,1)
        self.fraud_pred = -1
        self.appeal

    def fraud_algo(self):
        rng = np.random.default_rng()
        # self.fraud_pred = rng.binomial(1, 0.5)
        self.fraud_pred = rng.binomial(1, 0.5*(1-self.wealth))

    def appeal(self):
        rng = np.random.default_rng()
        print('bye')
        if self.fraud_pred == 1 and self.wealth > 0.2:
            self.fraud_algo()
            # self.fraud_pred = rng.binomial(1, 0.4)


    def step(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        self.appeal()
        print("Hi, I am agent " + str(self.unique_id) + ".")
        # print("my wealth, job, fraud, fraud_pred is:" + str(self.wealth)+ str(self.job) + str(self.fraud)+ str(self.fraud_pred))


class AgentModel(mesa.Model):

    """A model with some number of agents."""

    def __init__(self, N):
        self.num_hagents = N
        self.num_AIagents = 1
        self.schedule = mesa.time.RandomActivation(self)
        self.iteration = 0
        self.max_iters = 100

        
        # Create Human agents
        for i in range(self.num_hagents):
            a = Agent(i, self)
            a.fraud_algo()
            # a.appeal()
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(          
            agent_reporters={"unique_id": "unique_id", "wealth":"wealth","job":"job", "fraud":"fraud", "fraud_pred":"fraud_pred" })



    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

def tp(model_out):
    df = model_out.reset_index(level = 'AgentID')
    # df = pd.DataFrame(model_out,columns=["fraud","fraud_pred", "wealth"])
    df['tp'] = df['fraud'].eq(df['fraud_pred'])
    df['fp'] = df.apply(lambda x: 1 if x['fraud'] == 0 and x['fraud_pred'] == 1
                     else 0, axis=1)

    for i in (np.unique(df.index.get_level_values(0))):
        x = (df.loc[(df.index.get_level_values('Step')==i)])
        df.loc[(i,slice(None)),'wealth_perc'] = x['wealth'].rank(pct=True).astype(float)
        # df.loc[(i,slice(None)),'wealth_perc'] 

    print(df)

    # print(x/x.sum())
    # for date, new_df in df.groupby(level=0):
    #     print(new_df)

    return df

    


def viz(df):
    # courses = model_out['fraud_pred']
    # values = model_out['wealth']
    # print(courses)
    # df = pd.DataFrame(model_out,columns=["fraud","fraud_pred", "wealth"])
    # df['fraud_pred'] = df['fraud_pred'].astype(float)
    # df['wealth'] = df['wealth'].astype(float)
    # df['fraud'] = df['fraud'].astype(float)

    # sns.lineplot(data= df, x= "Step", y = "fp")
    # df2['tp_count'] = np.nan
    df50 = df[df['wealth_perc']<0.5]
    df50p = df[df['wealth_perc']>=0.5]

    for i in (np.unique(df.index.get_level_values(0))):
        x = (df.loc[(df.index.get_level_values('Step')==i)])
        df50.loc[(i,slice(None)),'fp_count'] = x['fp'].sum().astype(float)
        df50p.loc[(i,slice(None)),'fp_count'] = x['fp'].sum().astype(float)


    # fig, ax = plt.subplots()
    # ax.plot(df.index, )
    df50 = (df50.unstack(level =1)) #.plot(y = 'fp_count', kind='line')
    ax = df50p.plot(x = 'Step', y = 'fp_count', kind='line')
    df50.plot(ax=ax, y = 'fp_count',)

    # plt.plot
    print('hi')
    print(df50p)
    plt.show()
    
    #have a plot over time with <0.5 sum fp





if __name__ == "__main__":

    # np.random.seed(42)
    empty_model = AgentModel(100)
    for i in range(100):
        empty_model.step()
    model_out = empty_model.datacollector.get_agent_vars_dataframe()
    df2 = tp(model_out)
    # print(df2.reset_index(level = 'AgentID', drop =))
    print(df2)
    print(model_out)
    viz(df2)

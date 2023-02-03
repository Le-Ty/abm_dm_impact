import mesa
import numpy as np



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

    def fraud_algo(self):
        rng = np.random.default_rng()
        # if self.unique_id % 2 == 1:
        #     self.fraud_pred = 0
        # else:
        #     self.fraud_pred = 1
        self.fraud_pred = rng.binomial(1, 0.5*(1-self.wealth))
        print(0.5*(1-self.wealth))
        print(self.fraud_pred)

    def appeal(self):
        rng = np.random.default_rng()
        if self.fraud_pred == 1 and self.wealth > 0.1:
            # self.fraud_pred = [0]
            self.fraud_algo()
            print('hi')


    def step(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
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
            a.appeal()
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(          
            agent_reporters={"unique_id": "unique_id", "wealth":"wealth","job":"job", "fraud":"fraud", "fraud_pred":"fraud_pred" })



    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        # self.iteration += 1
        # if self.iteration > self.max_iters:
        #     self.running = False


def viz(model_out):
    ax = model_out.plot()
    ax.set_title("Distribution")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Citizens")
    _ = ax.legend(bbox_to_anchor=(1.35, 1.025))


if __name__ == "__main__":

    # np.random.seed(42)
    empty_model = AgentModel(100)
    for i in range(4):
        empty_model.step()
    model_out = empty_model.datacollector.get_agent_vars_dataframe()
    print(model_out)
    viz(model_out)
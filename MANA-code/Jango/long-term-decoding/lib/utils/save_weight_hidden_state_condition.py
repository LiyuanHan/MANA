
# decoding model weight and hidden state save conditions

save_weight_hidden_state__rel_path = "saved_weight_hidden_state"

save_weight_hidden_state__domain = [0, 4, 5]
    
save_whs__test_days = list(range(1, 120))
save_whs__turns = [0]
save_whs__epoch_step = 10       # total epochs is usually 150

class save_weight_hidden_state__condition:
    def __init__(self, test_days=save_whs__test_days, turns=save_whs__turns, epoch_step=save_whs__epoch_step):
        self.test_days = test_days
        self.turns = turns
        self.epoch_step = epoch_step
        pass

    def check_test_day(self, test_day):
        return (test_day in self.test_days)
    
    def check_turn(self, turn):
        return (turn in self.turns)
    
    def check_epoch(self, epoch):
        return ((epoch + 1) % self.epoch_step == 0)


# def save_weight_hidden_state__condition(test_day, turn, epoch):
    
#     return ((test_day in save_whs__testDays) and (turn == 0) and ((epoch + 1) % save_whs__epoch_step == 0))
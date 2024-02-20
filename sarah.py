'''
Sarah Chen
Homework 5
February 15, 2023
CS 171-061
'''

# function 1
def sumDigits(number): # number is passed from function 2
    # initialize sumAdd as 0  
    sumAdd = 0
    # for loop to add every element in number list together
    for num in number:
        sumAdd += num
    # if statement for sumAdd conditions
    if sumAdd > 0 and sumAdd < 10:
        return sumAdd # return sumAdd if true
    else: # Divide by 10 if sumAdd is greater than 9
        while sumAdd >= 10: # while loop to repeat floor division for big numbers
            HunDigit = sumAdd // 10     # Get the hundredth digit
            TenthDigit = sumAdd % 10    # Get the tenth digit
            sumAdd = HunDigit + TenthDigit  # Add them together
        return sumAdd

# function 2
def nameNumber(name):
    # initialize numList as an empty list
    number = []
    # for loop to iterate through each character of name
    # use .upper so that lowercase inputs are converted to uppercase
    for i in name.upper():
        # use .isalnum() to ignore special characters
        if i.isalnum():
            # if statements to see if the letters in the input meets condition and append value to number
            if i == "A" or i == "I" or i == "J" or i == "Q" or i == "Y":
                number.append(1)
            elif i == "B" or i == "K" or i == "R":
                number.append(2)
            elif i == "C" or i == "G" or i == "L" or i == "S":
                number.append(3)
            elif i == "D" or i == "M" or i == "T":
                number.append(4)
            elif i == "E" or i == "H" or i == "N" or i == "X":
                number.append(5)
            elif i == "U" or i == "V" or i == "W":
                number.append(6)
            elif i == "O" or i == "Z":
                number.append(7)
            elif i == "F" or i == "P":
                number.append(8)
    return sumDigits(number)
      
# function 3
def prediction(number): # number is passed as a sum of number in function 2
    # if statements to see if number from function 2 corresponds to condition
    if number == 1:
        return "A person who is successful in personal ambitions."
    elif number == 2:
        return "A gentle and artistic person."
    elif number == 3:
        return "A success in their professional career."
    elif number == 4:
        return "An unlucky person who must put in extra work for success."
    elif number == 5:
        return "A lucky person who leads an unconventional life."
    elif number == 6:
        return "A person who commands the respect of others."
    elif number == 7:
        return "A person who has a strong inner spirit."
    elif number == 8:
        return "A person who is misunderstood by others and is not respected for their success."
    elif number == 9:
        return "A person who is more successful in matters of the material than spiritual."
    else:
        return "Invalid Input"

# Main function
if __name__ == "__main__":
    # intro message for the generator
    print("Welcome to Name Number Generator")
    # ask user to enter their name
    name = input("Enter Your Name: ")
    
    # assign function 2 to variable for easier reference
    userNameNum = nameNumber(name)
    print(f"Your Name Number is {userNameNum}")
    
    # assign function 3 to variable for easier reference
    # use userNameNum instead of number since the latter is located within function 2
    predictionOutput = prediction(userNameNum)
    print(f"We predict you are: \n{predictionOutput}")
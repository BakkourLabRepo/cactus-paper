import instructions from "@jspsych/plugin-instructions";

const introInstructions = (dim) => {
  if (dim === 'line') {
    dim = 'angle'
  }
  return {
  type: instructions,
  pages: [
    "Welcome to the experiment.<br> You will be asked to perform a series of tasks today.<br>",
    "In these tasks you will be learning about the values of different shapes, colors, and angles.<br>",
    "Using these values you will then make a series of decisions<br> to earn points that will be converted into a bonus payment.<br>",
    `Let's start off with the first task<br> You will initially see a video of an item that is slowly changing its ${dim}.<br>`,
    `After watching this video, you will be asked to make decisions<br> about which ${dim} you think is a higher or lower value.<br><br>` +
      "You can press 'f' to select the item on the left, and 'j' to select the item on the right.<br>" +
      "You will receive feedback showing you the values of both items.<br> The value of the other item could be higher or lower.<br>" +
      "You will be shown the value above each item.<br> If you respond correctly to the prompt, the number above your choice will be in green, if you respond incorrectly it will be red.<br>",
    "Make sure you pay close attention to the question at the bottom as it will ask you for higher <b>or</b> lower values.<br>" +
    "When you are done with this task,<br>you will receive instructions for another task.<br><br>Press 'j' when you are ready.<br>",
  ],
  show_clickable_nav: true,
  key_forward: "j",
}};

const preCategoryInstructions = {
  type: instructions,
  pages: [
    "Great job!<br><br>Now you will be doing a different task where you will see a box on your screen with some of the items that you just learned values for alongside it.<br>",
    "You will be asked to place these items in any positions in that box that you see fit. You will do this twice, the first time you are asked to sort based on how similar things look.<br>" +
    "Then you will be asked to sort the items based on their values.<br>",
    "You will have a minute to move things around and then hit the submit button.<br><br>" +
      "You will not receive any feedback for this task but try and do your best.<br><br>Press 'j' when you are ready.<br>",
  ],
  show_clickable_nav: true,
  key_forward: "j",
};

const preChoiceInstructions1 = {
  type: instructions,
  pages: [
    "Great job! You are now ready for the final task.<br> This is the one where you can earn a bonus!<br>",
    "In this task you will see two stimuli that are made up of a color, a shape, and an angle.<br><br>" +
      "Here is an example image: <br>" +
      '<img src="example_choice_trial.png"></img>',
    "In this task, the screen will be blurred out. You will have to move your mouse around the screen to uncover the identity of the stimuli.<br>" +
      "Once you have looked at all the items, you can either choose the column on the left, or the column on the right.<br>",
    "You can click on the button labelled 'left' to choose the set of items on the left, and the button labelled 'right' to choose those on the right.<br>" +
      "You will have as much time as you need to move your spotlight from item to item before making your decision.<br>" +
      "We will be giving you a practice so you can see what moving the spotlight is like.<br><br>Press 'j' when you are ready.<br>",
  ],
  show_clickable_nav: true,
  key_forward: "j",
};

var preChoiceInstructions2 = {
  type: jsPsychInstructions,
  pages: [
    "Great job! You are now ready for the actual choice task.<br>Remember, this is the one where you can earn a bonus!<br>",
    "In this task you will see two stimuli that are made up of a color, a shape, and an angle.<br><br>" + "Here is an example image: <br>" + '<img src="example_choice_trial.png"></img>',
    "In this task, the screen will flash and then be blurred out. You will have to move your mouse around the screen to uncover the identity of the stimuli.<br>" + "Once you have looked at all the items, you can either choose the column on the left, or the column on the right.<br>",
    "You can click on the button labelled 'left' to choose the set of items on the left, and the button labelled 'right' to choose those on the right.<br>" + "You will have up to 2 minutes per trial to move your spotlight from item to item before making your decision.<br>" + "You will not receive feedback showing you the value of the item you selected. But we are tallying your points for your bonus!<br><br>There will be 120 trials of this task with a break in the middle, press 'j' when you are ready.<br>"
  ],
  show_clickable_nav: true,
  key_forward: "j"
};
const postCat1Instructions = (prev_dim, curr_dim) => {
  if (curr_dim === 'line') {
    curr_dim = 'angle'
  }
  if (prev_dim === 'line') {
    prev_dim = 'angle'
  }
  return {
  type: instructions,
  pages: [
    `Great job!<br><br>Now you will be doing the previous two tasks with the ${curr_dim} of an object, instead of ${prev_dim}.<br>`,
    `You will initially see a video of an item that is slowly changing its ${curr_dim}.<br>`,
    `After watching this video, you will be asked to make decisions<br> about which ${curr_dim} you think is a higher or lower value.<br><br>` +
      "You can press 'f' to select the item on the left, and 'j' to select the item on the right.<br>" +
      "You will receive feedback showing you the value of the item you selected.<br><br>Press 'j' when you are ready.<br>",
  ],
  show_clickable_nav: true,
  key_forward: "j",
}};



export {
  introInstructions,
  preCategoryInstructions,
  preChoiceInstructions1,
  preChoiceInstructions2,
  postCat1Instructions,
};

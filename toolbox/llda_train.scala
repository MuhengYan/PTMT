// Author: Muheng Yan
// Modified from "Stanford TMT Example 6 - Training a LabeledLDA model"
// http://nlp.stanford.edu/software/tmt/0.4/
// to take extra parameters for tuning models

/*
  Parameters:
  Para    Type  Name                  Description                                  
  ---------------------------------------------------------------------------------
  args(0) (Int) max Iteration         The max iteration you want lLDA to train     
  args(1) (Int) Term Length Filter    Minimum length of terms taken                
  args(2) (Int) Term MinDoc Filter    Minimum appearance of terms in docs taken    
  args(3) (Int) common Term Filter    Numbers of the most common terms removed     
  args(4) (Int) doc length Filter     Minimum length of Doc taken                    
  args(5) (Int) Label MinDoc Filter   Minimum appearance of labels in docs taken   
  args(6) (Str) Model Name            The name of the lLDA model trained
  args(7) (Str) Absolute Path         The absolute path to ".../PTMT/toolbox" 
  ---------------------------------------------------------------------------------
*/

// tells Scala where to find the TMT classes
import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

val source = CSVFile(args(7) + "train.csv") ~> IDColumn(1);

val tokenizer = {
  SimpleEnglishTokenizer() ~>                        // tokenize on space and punctuation
  CaseFolder() ~>                                    // lowercase everything
  WordsAndNumbersOnlyFilter() ~>                     // ignore non-words and non-numbers
  MinimumLengthFilter(args(1).toInt)                 // take terms with >= args(1) characters
}

val text = {
  source ~>                                          // read from the source file
  Column(3) ~>                                       // select column containing text
  TokenizeWith(tokenizer) ~>                         // tokenize with tokenizer above
  TermCounter() ~>                                   // collect counts (needed below)
  TermMinimumDocumentCountFilter(args(2).toInt) ~>   // filter terms in < args(2) docs
  TermDynamicStopListFilter(args(3).toInt) ~>        // filter out args(3) most common terms
  DocumentMinimumLengthFilter(args(4).toInt)         // take only docs with >= args(4) terms
}

// define fields from the dataset we are going to slice against
val labels = {
  source ~>                                          // read from the source file
  Column(2) ~>                                       // take column two, the year
  TokenizeWith(WhitespaceTokenizer()) ~>             // turns label field into an array
  TermCounter() ~>                                   // collect label counts
  TermMinimumDocumentCountFilter(args(5).toInt)      // filter labels in < args(5) docs
}

val dataset = LabeledLDADataset(text, labels);

// define the model parameters
val modelParams = LabeledLDAModelParams(dataset);

// Name of the output model folder to generate, Path = args(6)
val modelPath = file(args(6));

// Trains the model, writing to the given output path, MaxIter = args(0)
val model = TrainCVB0LabeledLDA(modelParams, dataset, output = modelPath, maxIterations = args(0).toInt);
// or could use TrainGibbsLabeledLDA(modelParams, dataset, output = modelPath, maxIterations = 1500);

// print the label index, or output as .csv file
for(i <- 0 to (model.numTopics - 1)){
  println("[TopicName] name of topic " + i + " : " + model.topicName(i));
}




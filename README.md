# PTMT

Stanford Topic Modeling Toolbox for Python. 

The training and inference processes can be execute separately as long as you have the correct model in your local file system.



# Installation
Please download Stanford Topic Modeling Toolbox along with this package, and put the .jar file under .../PTMT/toolbox

In terminal or cmd you can use command line to obtain the package. With linux systems you can copy the following codes:

```shell
cd ~
git clone https://github.com/MuhengYan/PTMT
cd ~/PTMT/toolbox
wget http://nlp.stanford.edu/software/tmt/tmt-0.4/tmt-0.4.0.jar
```

#Simple Example

```python
import PTMT
train_labels = ['candidate trump', 'candidate clinton', 'crime police', 'crime']
train_texts = ["Trump keeps saying Hillary's corrupt. Is he freaking serious?! He's the one with a whole resume of corruption.",
               "Clinton Foundation set up to: Enrich the Clintons; Sell access; rade political favors. Influence peddling",
               "El Cajon police kill 30-year-old black man Alfred Olango Police in El Cajon, California",
               "crime is bad. man should never commit crimes"]
true_labels = ['trump','candidate','police']
infer_texts = ["Donald Trump's company violated the U.S. embargo against Cuba",
               "trump and clinton are gonna have more debates",
               "there is a police arresting a man who committed a crime"]

model = PTMT.PythonTMT('trial')
model.train(train_labels, train_texts)
model.infer(infer_texts)
prec = model.evaluation(true_labels)
print("The precision of text inference is: ", prec)
```

If you have installed the package correctly, it would eventually shows "The precision of text inference is: 0.75"


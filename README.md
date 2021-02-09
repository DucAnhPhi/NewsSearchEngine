
- [TODOs](#todos)
- [Installation](#installation)
- [Run experiments](#getting-started)
- [Run unit tests](#run-unit-tests)
- [License](#license)
- [Author](#author)

## TODOs
- get strong baseline retrieval method with high recall
- implement LTR
- support more outlets
- handle scaling vector storage (max elements)
- implement webapp with async component and model server
- log queries and clicks for judgement list and popularity metric
- fix output path for scraper
- refactor netzpolitik indexing

## Installation
* Install Python 3.6.9 (if not already installed)

* On Windows: [Install Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)

* Install [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

* **Recommended:**
Setup a python virtual environment for this project. It keeps the dependencies required by different projects in separate places.

```
$ pip install virtualenv

$ cd LinguisticAnalysis

$ virtualenv -p python3.6.9 env
```
* To begin using the virtual environment, it needs to be activated:

```
$ source env/bin/activate
```

* Finally install all dependencies running:

```
$ pip install -r requirements.txt
```

* If you are done working in the virtual environment for the moment, you can deactivate it (remember to activate it again on usage):

```
$ deactivate
```

* To delete a virtual environment, just delete its folder.

```
rm -rf env
```

### Speed up performance

You can speed up the encoding of embeddings if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with atleast 2GB of free video memory:
- Install [CUDA](https://developer.nvidia.com/cuda-downloads)


## Run experiments




### Download Datasets

The following datasets were used to evaluate all experiments in this repository. Please download the following datasets if you want to reproduce our experiments:

- To reproduce our experiments with the *TREC Washington Post Corpus (WAPO)*, request the dataset [here](https://trec.nist.gov/data/wapost/) and store the **.jl** file in the **data/** folder
- To reproduce our experiments with *netzpolitik.org* run the following command in the parent directory of the project, to scrape all articles from 2012 to 2020 (the output is going to be stored in **data/netzpolitik.jsonl**):
```
python -m NewsSearchEngine.netzpolitik.scraper
```

### Index Datasets

We provide indexing scripts for both datasets. Go to the parent directory of this project and run the following commands:

```
python -m NewsSearchEngine.wapo.index_es
python -m NewsSearchEngine.wapo.index_vs
```


```
python -m NewsSearchEngine.netzpolitik.index_es
```

### Run experiment scripts

```
python -m NewsSearchEngine.wapo.experiments.keyword_match_recall
```

```
python -m NewsSearchEngine.netzpolitik.experiments.keyword_match_recall

python -m NewsSearchEngine.netzpolitik.experiments.semantic_search_recall

python -m NewsSearchEngine.netzpolitik.experiments.combined_recall
```

### TREC Washington Post document collection (WAPO)

The TREC Washington Post Corpus contains 671,947 news articles and blog posts from January 2012 through December 2019. You can request this data [here](https://trec.nist.gov/data/wapost/).

For the background linking task in 2018 (TREC 2018), TREC provided:

- [50 test topics for background linking task](https://trec.nist.gov/data/news/2018/newsir18-topics.txt)
- [relevance judgments for backgroundlinking task (with exponential gain values)](https://trec.nist.gov/data/news/2018/bqrels.exp-gains.txt)

We put both sets together in a more usable [JSON Lines Format](https://jsonlines.org/) in **data/judgement_list_wapo**.

According to the [TREC 2020 News Track Guidelines](http://trec-news.org/guidelines-2020.pdf) we removed articles from the dataset which are labeled in the "kicker" field as "Opinion", "Letters to the Editor", or "The Post's View", as they are **not relevant**. Additionally we removed articles which are labeled in the "kicker" field as "Test" as they contain "Lorem ipsum" text, thus being irrelevant aswell.
After the filtering only 487,322 news articles remain, which is a decrease in size by 27%.

It is important to note, that 5 articles listed in the [50 test topics for background linking task](https://trec.nist.gov/data/news/2018/newsir18-topics.txt) are missing in the dataset:

```
# available
<top>
<num> Number: 321 </num>
<docid>9171debc316e5e2782e0d2404ca7d09d</docid>
<url>https://www.washingtonpost.com/news/worldviews/wp/2016/09/01/women-are-half-of-the-world-but-only-22-percent-of-its-parliaments/<url>
</top>

# not available
<top>
<num> Number: 823 </num>
<docid>c109cc839f2d2414251471c48ae5515c</docid>
<url>https://www.washingtonpost.com/news/to-your-health/wp/2016/09/21/superbug-mrsa-may-be-spreading-through-tainted-poultry/2695269178/</url>
</top>

# available
<num> Number: 812 </num>
<docid>dcd1560bd13a0b665b95d3ba27cc960c</docid>
<url>https://www.washingtonpost.com/news/morning-mix/wp/2016/05/23/after-years-of-alleged-bullying-an-ohio-teen-killed-herself-is-her-school-district-responsible/<url>

# available
<num> Number: 811 </num>
<docid>a244d1e0cfd916a2af76b6a6c785b017</docid>
<url>https://www.washingtonpost.com/news/morning-mix/wp/2015/07/22/car-hacking-just-got-real-hackers-disable-suv-on-busy-highway/</url>

# not available
<num> Number: 803 </num>
<docid>cad56e871cd0bca6cc77e97ffe246258</docid>
<url>https://www.washingtonpost.com/news/wonk/wp/2016/05/11/the-middle-class-is-shrinking-just-about-everywhere-in-america/4/</url>
```

You could add the 3 of the missing but available articles manually in elasticsearch:

```
PUT /wapo_clean/_create/a244d1e0cfd916a2af76b6a6c785b017
{
    "title": "‘Car hacking’ just got real: In experiment, hackers disable SUV on busy highway",
    "offset_first_paragraph": 34,
    "date": 1437560700000,
    "kicker": "Morning Mix",
    "author": ["Michael E. Miller"],
    "text": "It was a driver\u2019s worst nightmare.\r\nAndy Greenberg was speeding along a busy interstate in St. Louis recently when he suddenly lost control of his vehicle. The accelerator abruptly stopped working. The car crawled to a stop. As 18-wheelers whizzed by his stalled vehicle, Greenberg began to panic.\r\nHis car hadn\u2019t spun out on black ice, however. It hadn\u2019t been hit by another vehicle or experienced engine trouble.\r\n\r\nIt had been hacked.\r\n\r\n[Hacks on the highway: Automakers rush to add wireless features, leaving our cars open to hackers]\r\n\r\nGreenberg, a senior writer for Wired magazine, had asked Charlie Miller and Chris Valasek \u2014 two \u201Cwhite hat\u201D or altruistic hackers \u2014 to show him what they could do.\r\n\r\nSo, while Greenberg drove down the highway, Miller and Valasek sat on Miller\u2019s couch 10 miles away and played God.\r\n\r\n\u201CThough I hadn\u2019t touched the dashboard, the vents in the Jeep Cherokee started blasting cold air at the maximum setting, chilling the sweat on my back through the in-seat climate control system,\u201D Greenberg wrote. \u201CNext the radio switched to the local hip hop station and began blaring Skee-lo at full volume. I spun the control knob left and hit the power button, to no avail. Then the windshield wipers turned on, and wiper fluid blurred the glass.\r\n\r\nAD\r\n\r\n\u201CAs I tried to cope with all this, a picture of the two hackers performing these stunts appeared on the car\u2019s digital display: Charlie Miller and Chris Valasek, wearing their trademark track suits. A nice touch, I thought.\u201D\r\n\r\nThe situation stopped being funny, however, when the two hackers cut the engine.\r\n\r\n\u201CSeriously, this is f\u2014\u2013 dangerous. I need to move,\u201D Greenberg said, pleading for the hackers to return power to the vehicle.\r\n\r\nThe 2014 Jeep Cherokee undergoes assembly at the Chrysler Toledo North Assembly Plant Jeep May 7, 2014 in Toledo, Ohio.\r\nGreenberg survived to tell his tale, of course, but the ordeal is just the latest in a series of incidents highlighting the startling security vulnerabilities of hundreds of thousands of American automobiles.\r\n\r\nThese incidents have raised the specter of remote-controlled car accidents, in which anarchist hackers or computer-savvy assassins could still be at home in their pajamas while wreaking havoc.\r\n\r\nAD\r\n\r\nOn Tuesday, just hours after Wired published its story, Sens. Ed Markey (D-Mass.) and Richard Blumenthal (D-Conn.) unveiled a bill aimed at keeping Internet-connected cars from getting hacked.\r\n\r\n\u201CRushing to roll out the next big thing, automakers have left cars unlocked to hackers and data-trackers,\u201D Blumenthal said.\r\n\r\n\u201CControlled demonstrations show how frightening it would be to have a hacker take over controls of a car,\u201D Markey said in a statement to Wired. \u201CDrivers shouldn\u2019t have to choose between being connected and being protected\u2026We need clear rules of the road that protect cars from hackers and American families from data trackers.\u201D\r\n\r\n[Next dashboard warning may be, \u2018Your car has been hacked!\u2019]\r\n\r\nEven the hackers themselves were taken aback by their abilities.\r\n\r\n\u201CWhen I saw we could do it anywhere, over the Internet, I freaked out,\u201D Valasek told Wired. \u201CI was frightened. It was like, holy f\u2014, that\u2019s a vehicle on a highway in the middle of the country. Car hacking got real, right then.\u201D\r\n\r\nAD\r\n\r\nThe problem is one of our own creation.\r\n\r\nLike thousands of other everyday devices, from coffeemakers to power plants, cars are increasingly connected to the Internet. This enables drivers to stream music, watch videos and use GPS.\r\n\r\nBut it also exposes their cars \u2014 and therefore the drivers as well \u2014 to hackers.\r\n\r\nMiller and Valasek exploited a weak spot in Uconnect, an Internet-connected feature on as many as 471,000 Fiat Chrysler late-model automobiles, most of them in the United States. Using a laptop computer and a burner phone, they were able to send a series of commands to the car.\r\n\r\n\u201CUconnect computers are linked to the Internet by Sprint\u2019s cellular network, and only other Sprint devices can talk to them,\u201D Greenberg explained. By connecting a phone to his laptop, Miller was able to use the phone as a Wi-Fi hot spot and search Sprint\u2019s entire 3G network for hack-able cars.\r\n\r\nAD\r\n\r\n[Hackers warned senators of the Internet\u2019s vulnerabilities back in 1998, but were ignored]\r\n\r\nNot only does the computer weakness allow hackers to manipulate the locks and turn off the engine, it also enables them to cut the brakes. They can even take over the steering wheel if the car is in reverse.\r\n\r\n\u201CFrom an attacker\u2019s perspective, it\u2019s a super nice vulnerability,\u201D Miller told Greenberg.\r\n\r\nThe stunt seems to confirm fears that have worried security experts for several years now. In 2011, researchers at the University of Washington and the University of California at San Diego proved they could remotely disable a car\u2019s locks and brakes.\r\n\r\nWhile the researchers didn\u2019t reveal the car manufacturer, Miller and Valasek have made no secret that their hack affects cars made by Fiat Chrysler.\r\n\r\nBefore going public with the news, however, the hackers took their findings to the company. Chrysler has recently released a patch to prevent such hacking.\r\n\r\n\r\n\u201C[Fiat Chrysler Automobiles] has a program in place to continuously test vehicles systems to identify vulnerabilities and develop solutions,\u201D the company said in a statement sent to WIRED. \u201CFCA is committed to providing customers with the latest software updates to secure vehicles against any potential vulnerability.\u201D\r\n\r\nAD\r\n\r\n\u201CPatch your Chrysler vehicle before hackers kill you,\u201D warned Fox News on Wednesday after Wired published its article.\r\n\r\nThanks to Miller and Valasek, Chrysler drivers can now guard against such invasions. But the Uconnect weakness is only the tip of an Internet security iceberg. There are many other ways that a car can be compromised by hackers.\r\n\r\nOther brands, for example, might not be any safer.\r\n\r\n\u201CI don\u2019t think there are qualitative differences in security between vehicles today,\u201D UCSD computer science professor Stefan Savage told Wired. \u201CThe Europeans are a little bit ahead. The Japanese are a little bit behind. But broadly writ, this is something everyone\u2019s still getting their hands around.\u201D\r\n\r\nIn February, hackers demonstrated to NBC 4 in New York how they could override a car\u2019s system using a tiny Wi-Fi dongle plugged underneath its steering wheel.\r\n\r\nAD\r\n\r\n[FBI probe of alleged plane hack sparks worries over flight safety]\r\n\r\nOther successful attacks have involved \u201Cinfecting the computers in the repair shop and then having that infection spread to the car through the diagnostic port, or hacking in through the Bluetooth system, or using the telematics unit that\u2019s normally used to provide roadside assistance,\u201D Kathleen Fisher from the federal Defense Advanced Research Projects Agency (DARPA), told NBC.\r\n\r\nCar makers have been slow to respond to criticism from researchers or hackers like Miller and Valasek.\r\n\r\n\u201CThere is a clear lack of appropriate security measures to protect drivers against hackers who may be able to take control of a vehicle or against those who may wish to collect and use personal driver information,\u201D according to a study compiled by Markey and released in February.\r\n\r\nAD\r\n\r\nThe study, \u201CTracking & Hacking: Security & Privacy Gaps Put American Drivers at Risk,\u201D found, among other things, that:\r\n\r\nNearly 100% of cars on the market include wireless technologies that could pose vulnerabilities to hacking or privacy intrusions.\r\nMost automobile manufacturers were unaware of or unable to report on past hacking incidents.\r\nSecurity measures to prevent remote access to vehicle electronics are inconsistent and haphazard across all automobile manufacturers, and many manufacturers did not seem to understand the questions posed by Senator Markey.\r\nOnly two automobile manufacturers were able to describe any capabilities to diagnose or meaningfully respond to an infiltration in real-time, and most say they rely on technologies that cannot be used for this purpose at all.\r\nThe security shortcomings exposed by Miller, Valasek and others are especially worrying as fully automated cars appear on the horizon.\r\n\r\nImagine laying back in your fully automated car on your way to work when someone at a Starbucks miles away takes control and sends your robotic car swerving into oncoming traffic.\r\n\r\n[The government push to regulate driverless cars has finally begun]\r\n\r\nIf you think that\u2019s scary, however, there are a countless other devices that could, theoretically, fall under the sway of hackers.\r\n\r\nA computer security advocacy group called I Am The Cavalry warns that the threat goes far beyond cars to include common Wi-Fi connected medical devices like IV pumps or implantable pacemakers, electronic home security systems, and \u2014 on a grander scale \u2014 public infrastructure like railways, airplanes and power plants.\r\n\r\nAD\r\n\r\n[Yes, terrorists could have hacked Dick Cheney\u2019s heart]\r\n\r\n\u201CWhen you get up in the morning and get in your car to go to work, by the time you\u2019ve gotten to work and sat down at your desk, you\u2019ve literally interacted with probably several hundred of those controllers from when you turn on the tap to brush your teeth, to when you turn on the power to when you turn on your car engine,\u201D Tom Parker, a professional hacker hired to help companies find their systems\u2019 flaws, told NBC 4.\r\n\r\nMiller and Valasek told Wired that they will give more details on their harrowing hack in two weeks at the annual Black Hat security conference in Las Vegas.\r\n\r\n\u201CThis is what everyone who thinks about car security has worried about for years,\u201D Miller told Greenberg. \u201CThis is a reality.\u201D",
    "links": ["http://www.washingtonpost.com/sf/business/2015/07/22/hacks-on-the-highway/?itid=lk_inline_manual_7", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "https://www.washingtonpost.com/blogs/the-switch/wp/2015/07/21/the-push-to-regulate-driverless-cars-has-finally-begun/?itid=lk_inline_manual_23", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "http://www.washingtonpost.com/local/trafficandcommuting/next-dashboard-warning-may-be-your-car-has-been-hacked/2015/07/21/ef26e9f4-2fbd-11e5-8353-1215475949f4_story.html?itid=lk_inline_manual_25", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "http://www.washingtonpost.com/sf/business/2015/06/22/net-of-insecurity-part-3/?itid=lk_inline_manual_38", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "http://www.autosec.org/pubs/cars-usenixsec2011.pdf", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "http://www.foxnews.com/tech/2015/07/21/patch-your-chrysler-vehicle-before-hackers-kill/", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/", "http://www.nbcnewyork.com/news/local/hack-a-car-computer-wifi-remote-vehicle-hacking-291272611.html", "http://www.washingtonpost.com/business/economy/fbi-probe-of-plane-hack-sparks-worries-over-flight-safety/2015/05/18/8f75e662-fd69-11e4-805c-c3f407e5a9e9_story.html?itid=lk_inline_manual_56", "http://www.nbcnewyork.com/news/local/hack-a-car-computer-wifi-remote-vehicle-hacking-291272611.html", "http://www.markey.senate.gov/imo/media/doc/2015-02-06_MarkeyReport-Tracking_Hacking_CarSecurity%202.pdf", "https://www.washingtonpost.com/blogs/the-switch/wp/2015/07/21/the-push-to-regulate-driverless-cars-has-finally-begun/?itid=lk_inline_manual_66", "https://www.iamthecavalry.org/about/overview/", "https://www.washingtonpost.com/blogs/the-switch/wp/2013/10/21/yes-terrorists-could-have-hacked-dick-cheneys-heart/?itid=lk_inline_manual_70", "http://www.nbcnewyork.com/news/local/hack-a-car-computer-wifi-remote-vehicle-hacking-291272611.html", "https://www.blackhat.com/us-15/registration.html", "http://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/"],
    "url": "https://www.washingtonpost.com/news/morning-mix/wp/2015/07/22/car-hacking-just-got-real-hackers-disable-suv-on-busy-highway/"
}

PUT /wapo_clean/_create/dcd1560bd13a0b665b95d3ba27cc960c
{
    "title": "After years of alleged bullying, an Ohio teen killed herself. Is her school district responsible?",
    "offset_first_paragraph": 241,
    "date": 1464004440000,
    "kicker": "Morning Mix",
    "author": ["Yanan Wang"],
    "text": "Growing up, Emilie Olsen had an infectious smile, a love for horses and a perfect attendance record. She was a straight-A student and an excellent volleyball player. Emilie \u201Chad an extremely sweet spirit about her,\u201D a family friend recalled.\r\n\r\nOn Dec. 11, 2014, the 13-year-old shot and killed herself at home.\r\n\r\nIt was a tragedy that sent a jolt through Fairfield, Ohio, where Emilie had lived since her parents, Marc and Cindy Olsen, adopted her from China when she was 9 months old. Classmates and neighbors mourned a young life cut short.\r\n\r\nBut in the days following Emilie\u2019s death, her parents spoke out against the seeming suddenness of it all. Emilie\u2019s death was precipitated by cruel, relentless bullying, the Olsens said. Worse: It could have been prevented, they claimed, if officials at Fairfield Intermediate and Middle School had been more responsive. Emilie did not have to die, they said.\r\n\r\nThese were the allegations made in an 82-page federal lawsuit the Olsens filed against Fairfield City School District, various administrators and Emilie\u2019s alleged bullies last December. Since then, their fight to hold the school district accountable for their daughter\u2019s death has been met with support from parents and denials from school officials.\r\n\r\nFormer Fairfield City Schools superintendent Paul Otten, who is among the defendants named in the complaint, left his position last month to become the superintendent of the nearby Beavercreek City School District.\r\n\r\nJust last week, Fairfield Middle School Principal Lincoln Butts, also a defendant, resigned for \u201Cpersonal reasons.\u201D\r\n\r\nHow to talk to your teen about depression and suicide\r\n\r\nNeither administrator gave specific explanations for leaving. In response to the lawsuit\u2019s December filing, the school district said in a statement, according to WCPO: \u201CThe District will be defending the litigation and will be providing appropriate responses in the course of the litigation. The District has no further comment at this time regarding this pending matter.\u201D\r\n\r\nMeanwhile, the lawsuit alleges that Emilie suffered discrimination because of her race and perceived sexual orientation and that school officials were negligent in their handling of her bullying. The complaint details a downward spiral brought on by unabating abuse.\r\n\r\nEscalating incidents\r\n\r\nIt started in the fifth grade, according to the complaint, when Emilie took to wearing camouflage-patterned clothing and cowboy boots. Her style allegedly prompted jeers from classmates, who called her \u201Cfake country\u201D \u2014 because \u201CChinese people don\u2019t wear camo.\u201D\r\n\r\nThings got worse when Emilie entered the sixth grade, the complaint said. She allegedly became the target of mean-spirited social media messages, as well as a fake Instagram account called \u201CEmilie Olsen is Gay.\u201D One classmate allegedly followed Emilie into the bathroom, handed her a razor and instructed her to \u201Cend her life.\u201D\r\n\r\nOther Instagram accounts surfaced, making sexually explicit comments and derogatory remarks about Emilie\u2019s perceived sexual orientation.\r\n\r\nIn the gym one day, Emilie and another female student got into a scuffle over the fake accounts, and the student allegedly pushed Emilie and slapped her in the face. A teacher allegedly witnessed and broke up the fight but took no action other than to direct the students \u201Cback to class.\u201D\r\n\r\n\u201CI have a bad feeling that if nothing is done then this has the possibility to escalate into something worse,\u201D Marc Olsen wrote in an email to the school\u2019s assistant principals after learning of the fight from the father of a student who saw it happen. He then received a phone call from one of the principals, who allegedly said they were \u201Cgoing to take care of the situation.\u201D\r\n\r\nAccording to the lawsuit, no students were ultimately disciplined for the fight or the Instagram accounts. Otten, the former superintendent, told WCPO last May that he didn\u2019t know about the online bullying.\r\n\r\n\u201CYou\u2019re assuming that when we get something (from parents) we start researching kids\u2019 Instagram accounts and printing stuff off,\u201D he said. \u201CI wasn\u2019t aware of it.\u201D\r\n\r\nOtten added that he has four kids enrolled in the school district. \u201CI can assure you,\u201D he told WCPO, \u201C\u2026 there is nothing for me to gain by hiding anything.\u201D\r\n\r\nThe bullying only intensified in seventh grade, the complaint said, when Emilie was placed in the same learning group as several girls who had allegedly harassed her the year before. By then, Emilie had started to inflict self-harm and express suicidal and depressive thoughts.\r\n\r\nWhen the Olsens brought these circumstances to the principals\u2019 attention, they were allegedly told that Emilie \u201Cneeded to buckle down\u201D and cope even though she told an administrator she was \u201Cfrightened to return to school.\u201D\r\n\r\nWhile the cyber-bullying continued, physical messages allegedly started to appear on the stalls and walls of the school\u2019s bathrooms. These scrawled scripts singled Emilie out by name \u2014 \u201CGo kill yourself Emilie\u201D \u2014 and made reference to Emilie\u2019s race and perceived sexual orientation. According to the complaint, while the messages were in \u201Ceasily observable\u201D locations and \u201Ccould not have been missed by anyone using the restrooms, including the school administrators\u201D and teachers, the staff failed to remove them in a timely manner.\r\n\r\nFive myths about suicide\r\n\r\nIn October 2014, a group of Emilie\u2019s friends defended her against her bullies in the cafeteria, initiating a verbal dispute as her friends yelled at the bullies to stop \u201Cmessing\u201D with Emilie. (The students\u2019 incident reports were obtained by WCPO.)\r\n\r\nAt a previously scheduled meeting with the assistant principals the next day, Olsen was not told about the fight involving his daughter, the complaint alleges. In the following days, Emilie started vomiting and feeling unwell. When Olsen informed the school that she would be absent because she was feeling sick, he was allegedly still not told about the fight.\r\n\r\nThrough all of this, Emilie was becoming increasingly withdrawn, barely recognizable from her old self, the complaint says. She seemed to take little interest in her schoolwork, and her grades dropped dramatically. On a personality quiz that Emilie was required to take for class, she described her \u201Cbad day symptoms\u201D as \u201Ccrying, depressing, yelling and screaming, passive resistance, and going into a trance,\u201D the complaint says.\r\n\r\nHer Internet search history showed attempts to get help, followed by growing despair, her parents say in their lawsuit.\r\n\r\nEmilie asked strangers online whether they had ever been bullied and viewed articles about celebrities who were bullied in school. She visited a website with the line \u201CI\u2019m just a kid and my life is a nightmare,\u201D and a picture of a young woman\u2019s slashed forearm, with the caption \u201CI\u2019m not strong anymore.\u201D\r\n\r\nAt a school where Asian Americans were allegedly labeled \u201Ctoo smart\u201D and mocked for their \u201Cslanted eyes,\u201D Emilie seemed to feel alienated, the complaint says.\r\n\r\nShe sought permission to dye her hair to \u201Clook more like a white person.\u201D Her parents allege that she asked her father, \u201CWhy can\u2019t I be white like you and mom?\u201D\r\n\r\nMarc Olsen later told police that Emilie had been suffering from depression and had a history of cutting herself, the Butler County Journal-News reported.\r\n\r\n\u201CI[\u2018m] causing all this trouble on earth,\u201D Emilie wrote to a friend on Facebook on Dec. 1, 2014, the complaint says. \u201CIt hurts when you have to explain yourself to people you don\u2019t know or like. You feel them judging you, staring at you, talking about you, and I\u2019ve made up my mind, I wanna die.\u201D\r\n\r\nShe went on: \u201CEven if there [sic] adults they hate me. I can\u2019t please anyone with anything I do. Not even my teachers.\u201D\r\n\r\nLess than two weeks later, Emilie killed herself with her father\u2019s gun.\r\n\r\nSo many people attempted suicide in this community that a state emergency is declared\r\n\r\nA school\u2019s role\r\n\r\n\r\nFive days after the suicide, the Olsens were in the throes of grief when they allegedly received a visit from Principal Butts and a group of police officers. According to the lawsuit, the officers coerced the Olsens to let them inside their home, then told Marc Olsen that he was \u201Cstirring the pot\u201D and \u201Centertaining rumors\u201D by talking to the media about Emilie\u2019s death.\r\n\r\nAt that point, Olsen had already told the Journal-News that he thought bullying killed Emilie.\r\n\r\nLater that month, Otten sent a letter to parents across the district addressing \u201Cthe many rumors and false reports which have surfaced\u201D about Emilie\u2019s passing.\r\n\r\n\u201CThe Fairfield Township Police Department conducted a thorough investigation and did not find any credible evidence that bullying was a factor in this tragedy,\u201D the letter said. \u201CIt seems there is an unjustified need to place blame for this horrendous event. I want you to know we all mourn the loss of Emilie. \u2026 However, the rumors and misinformation regarding this event which are being conveyed by social and other media are negatively affecting our community, our schools and our staff.\u201D\r\n\r\nA police report obtained by the Cincinnati Enquirer showed that authorities spoke to Emilie\u2019s boyfriend and best friend after her death. Neither mentioned bullying.\r\n\r\nSchool officials have acknowledged communication with the Olsens about the bullying but said the issues were resolved \u201Cto the complete satisfaction of the family.\u201D\r\n\r\nA jury trial for the suit is slated to take place in 2018.\r\n\r\nUnder Ohio law, bullying and cyberbullying are prohibited in schools. But families seeking redress face an uphill battle, the Journal-News reported, because they must prove that schools were aware of and deliberately indifferent to threats of harm.\r\n\r\n\u201CMaybe most importantly, did the school district know about [the bullying] and did they, as the plaintiffs allege here, literally put her back into classes with the bullies that they\u2019d reported about?\u201D said ABC News legal analyst Dan Abrams. \u201CNow the superintendent actually wrote a letter to the school community saying that he didn\u2019t think bullying had to do with this, so there\u2019s going to be clear factual disputes.\u201D\r\n\r\nA response filed by the defendants in April argues that the lawsuit\u2019s claims are invalid because school officials are not liable for violations of the anti-bullying statute. It further alleges that the school district (as opposed to the school board, which is also named) is not a valid party to the lawsuit because it is \u201Cmerely a geographical area.\u201D\r\n\r\nThe Olsens\u2019 lawsuit cites other alleged incidents of bullying at Fairfield City Schools, noting that their daughter\u2019s case \u201Cwas not an outlier.\u201D\r\n\r\n\u201CI\u2019m speaking up for [Emilie] because she can\u2019t do that now,\u201D Marc Olsen told the Journal-News shortly after her death. \u201CShe\u2019s over in a funeral home in downtown Hamilton. I need to do this. This is my job. I\u2019m not going to fail her again.\u201D\r\n\r\nThis post has been updated to clarify that the gun Emilie used belonged to her father.",
    "links": ["http://www.journal-news.com/news/news/father-claims-daughters-death-result-of-bullying/njSty/", "https://assets.documentcloud.org/documents/2646255/Fairfield-Wrongful-Death-Lawsuit.pdf", "http://www.wcpo.com/news/local-news/hamilton-county/fairfield/fairfield-city-schools-superintendent-paul-otten-to-take-reins-of-beavercreek-city-schools", "http://www.journal-news.com/news/news/local/fairfield-principal-accused-in-emilie-olsen-lawsui/nrPdY/", "https://www.washingtonpost.com/news/parenting/wp/2015/05/18/how-to-talk-to-your-teen-about-depression-suicide/?itid=lk_interstitial_manual_12", "http://www.wcpo.com/news/local-news/hamilton-county/fairfield/fairfield-city-schools-superintendent-paul-otten-to-take-reins-of-beavercreek-city-schools", "http://www.wcpo.com/longform/emilie-olsen-uncovered-evidence-shows-bullying-was-factor-in-13-year-old-suicide", "https://www.washingtonpost.com/opinions/five-myths-about-suicide/2016/05/06/5a537cbe-1236-11e6-81b4-581a5c4c42df_story.html?itid=lk_interstitial_manual_36", "https://s3.amazonaws.com/s3.documentcloud.org/documents/2081567/school-incident-reports-10-21-14.pdf", "http://www.journal-news.com/news/news/police-reports-shed-more-light-on-fairfield-teens-/njWtX/", "https://www.washingtonpost.com/news/morning-mix/wp/2016/04/13/so-many-people-attempted-suicide-in-this-community-that-a-state-of-emergency-is-declared/?itid=lk_interstitial_manual_54", "http://www.journal-news.com/news/news/father-claims-daughters-death-result-of-bullying/njSty/", "https://s3.amazonaws.com/s3.documentcloud.org/documents/2081568/otten-letter-to-community-12-19-14.pdf", "http://www.cincinnati.com/story/news/education/2015/12/14/emilie-olsens-family-sues-fairfield-schools-over-bullying-claims/77284500/", "http://www.journal-news.com/news/news/crime-law/expert-schools-often-prevail-in-bullying-lawsuits/npk93/", "http://abcnews.go.com/US/ohio-parents-allege-school-bullying-suit-teens-suicide/story?id=35793899", "http://www.journal-news.com/news/news/father-claims-daughters-death-result-of-bullying/njSty/"],
    "url": "https://www.washingtonpost.com/news/morning-mix/wp/2016/05/23/after-years-of-alleged-bullying-an-ohio-teen-killed-herself-is-her-school-district-responsible/"
}

PUT /wapo_clean/_create/9171debc316e5e2782e0d2404ca7d09d
{
    "title": "Despite a big year for women in politics, national legislatures are still dominated by men",
    "offset_first_paragraph": 38,
    "date": 1472720400000,
    "kicker": "WorldViews",
    "author": ["Melissa Etehad", "Jeremy C.F. Lin"],
    "text": "It's a big year for women in politics. In a historic first, Hillary Clinton was named the Democratic Party\u2019s presidential nominee in the upcoming U.S. elections. If she wins, she will join Theresa May of Britain and Angela Merkel of Germany in the ranks of women who lead prominent Western democracies. They're not alone, either. In recent years, the number of women holding positions in both parliament and executive government has grown. As of June 2016, women\u2019s membership in parliament doubled from 11.3 percent in 1995 to 22.1 percent in 2015, according to a recent study by the Inter-Parliamentary Union. And this year alone, there have been many historic firsts. In Iran, women won 17 seats during the Islamic Republic\u2019s parliamentary elections in February. \u201CThis is a record and we are happy that our dear women are taking part in all stages, especially in politics,\u201D Iranian President Hassan Rouhani said in a speech back in May. Africa has had some of the most dramatic breakthroughs over the past 20 years. As of June, four African nations were part of the top 10 as far as the number of women represented in parliament. Rwanda, for example, tops the chart as the country with the highest number of women represented in parliament in both the lower (63.8 percent) and upper houses (38.5 percent) in 2016. But women remain a relatively small group in parliaments around the world, and progress is slow. Data from the IPU shows that in a vast majority of countries, men still dominate the political stage. In both the lower and upper houses combined, women around the world account for around 22 percent of seats held in parliament. \"Yes it is an improvement considering that it started at a low base,\" Executive Director of U.N. Women Phumzile Mlambo Ngcuka said. \"But if we continue at this pace, it's going to take us too long. We need to fast-track the attainment of gender equality.\" The U.S. is one country that lags the global average of women in the national legislature. It ranks behind 95 countries, including low- and middle-income countries such as Ethiopia, El Salvador and Suriname. Other countries that outperform the United States include Iraq, Afghanistan and Saudi Arabia. One way that countries have been fast-tracking female participation in politics is through gender quotas, and the results have left many surprised. Countries that that have relatively low socioeconomic development or have just emerged from conflict, such as Afghanistan, Tanzania and Ecuador, have shown tremendous improvement and have higher levels of female representation in parliaments. Jennifer Rosen, assistant professor of sociology at Pepperdine University, says emerging democracies give leaders a unique opportunity for leaders to enshrine gender quotas in their new constitutions, helping bypass cultural barriers that would have otherwise stood in the way of women participating in politics for many years. Different electoral systems also influence women's participation in parliament. While scholars agree that proportional representation systems, which allow people to vote for a party list rather than a particular candidate, have a positive influence on female representation in government, some are divided as to whether it has a positive impact on all countries or only on Western democracies. One way for countries to shift their mentalities is to encourage people in executive leadership positions to set an example, Rosen said. Ngcuka, the U.N. women executive director, agrees and said men play an important role and that women shouldn't be the only ones trying to \"move the glass ceiling,\" noting how attitudes change when influential leaders such as Canadian Prime Minister Justin Trudeau nominated a cabinet in 2015 that was half-female. Political parties that are supposed to provide gender quotas need to ensure that they place women in winnable positions, IPU Secretary General Martin Chungong said. \"In many instances they are not walking the walk,\" he said. \"It's not enough to have quotas. We need to provide incentives for people who don't want to implement them.\" What's also clear is that having women in national legislatures can change the way a country is governed. Research shows that when women participate in politics, the conversation gets steered toward issues that their male counterparts often fail to address, such as family planning, education and gender-based violence. \"We have seen that there are certain issues women are better able to articulate,\" Chungong said. \"Women tend to take leading roles not because it is a women's issue, but because it has to do with livelihood of society as a whole.\"",
    "links": ["http://www.ipu.org/pdf/publications/WIP20Y-en.pdf", "http://iranprimer.usip.org/blog/2016/may/27/iran%E2%80%99s-runoff-election-parliament", "https://www.inclusivesecurity.org/wp-content/uploads/2013/05/Bringing-Women-into-Government_FINAL.pdf"],
    "url": "https://www.washingtonpost.com/news/worldviews/wp/2016/09/01/women-are-half-of-the-world-but-only-22-percent-of-its-parliaments/"
}
```

## Run unit tests

Run the following command in the current directory:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author

* **Phi, Duc Anh**
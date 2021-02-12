import pytest
import os
import json
from elasticsearch import Elasticsearch
from ...wapo.parser import ParserWAPO


class TestParserNetzpolitik():
    @classmethod
    def setup_class(self):
        self.es = Elasticsearch()
        self.parser = ParserWAPO(self.es)
        self.index = "wapo_clean"
        file_location = f"{os.path.abspath(os.path.join(__file__, os.pardir))}/test_articles_raw.jsonl"
        self.articles = []
        with open(file_location, "r", encoding="utf-8") as f:
            for line in f:
                self.articles.append(json.loads(line))

    def test_get_keywords_tf_idf(self):
        raw = self.articles[0]
        parsed = self.parser.parse_article(raw)
        expected_k = ["ahead", "commut", "region", "avenu", "beltwai", "bridg", "congest", "connector", "construct", "driver", "fare", "intercounti", "lane", "line", "mainten", "metro", "new", "open", "project", "rider", "road", "rout", "schedul", "station", "telegraph", "train", "transport", "travel"]
        actual_k = self.parser.get_keywords_tf_idf(self.index, raw["id"])
        intersection = list(set(expected_k) & set(actual_k))
        assert len(intersection) == len(expected_k)

    def test_parse_article_2012(self):
        raw = self.articles[0]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "The year ahead for D.C. region’s commuters"
        assert parsed["offset_first_paragraph"] == 281
        assert parsed["date"] == 1325444842000
        assert parsed["kicker"] == "Local"
        assert parsed["author"] == "Robert Thomson and Mark Berman"
        assert parsed["text"] == "It’s never enough, unless it’s too much. In 2012, commuters in the  D.C. region  will renew their love-hate relationship with transportation projects and programs, including some scheduled for completion and others just getting started. Here are 10 efforts likely to get attention. HOT lanes The  high-occupancy toll lanes  on the western side of the  Capital Beltway  are scheduled to open late in 2012. The D.C. region hasn’t seen anything like them. Will they become the way of the future? Travelers still ask about — and complain about — what’s going on in the 14-mile work zone between Springfield and the  Dulles Toll Road interchange . But they’ve also begun to ask how the lanes will function when they finally open. The HOT lanes managers will spend months preparing drivers to use them. And even before the lanes open, drivers will experience some improvements at the interchanges being rebuilt to accommodate the new lanes. Intercounty Connector After a half century of discussion and debate, opening 18 miles of the  Intercounty Connector  was a  top transportation story of 2011 . But it opened in segments, and the biggest part didn’t open till the end-of-the-year holidays were upon us. This year, we should see whether drivers really take to the new toll road or decide they will stick with the congestion and delay on the old routes. Many drivers probably will test out the connector and pick the portions of it that work for them under particular circumstances. Most times, it won’t be a question of paying $4 to use the entire highway at rush hour, but rather a choice to pay 70 cents to travel from southbound Interstate 95 to southbound Route 29, cutting a corner off the Capital Beltway when traffic reports say it’s especially congested. Beltway/Telegraph Road The repeated rounds of heavy rain this fall pushed back the  Woodrow Wilson Bridge project’s  goal of opening new lanes on the Capital Beltway near Telegraph Road in Virginia. Important parts of the remaining work on the Beltway require warmer weather, so expect to see the lanes in their current configuration through the winter. Then in late spring or early summer, a new portion of the THRU lanes will open in the zone between west of Route 1 and west of Telegraph Road. During the summer, the LOCAL lane segment also will be completed. This work will eliminate the three-lane bottleneck on the Beltway west of the Wilson Bridge, the obstacle that has prevented many drivers from enjoying the full benefits of the new, wider bridge. Federal base realignment More employees are scheduled to arrive at the  Mark Center , off Interstate 395 in Alexandria. Some changes have been made in the signal timings and lane markings nearby, but the main planned improvement is a new HOV ramp at I-395 and Seminary Road. The Virginia Department of Transportation has scheduled a public meeting on that project for Jan. 25. Meanwhile, the Maryland State Highway Administration will begin to upgrade intersections near  the newly consolidated Walter Reed National Military Medical Center  on Rockville Pike in Bethesda. Several projects are scheduled to start this spring. 11th Street Bridge This D.C. project also made the list of 2011’s top transportation stories, but several of the  new 11th Street Bridge’s  most important and beneficial elements aren’t scheduled to open till later this year. The new span taking traffic away from downtown and over the Anacostia River is scheduled to open this month, following December’s opening of the new inbound span. That will clear the way for completion of the ramps that will link the highways on either side of the river. Also scheduled for this year is completion of the third new span, which will provide a new link for local traffic between neighborhoods on both sides of the river. Metro map makeover At mid-year, Metrorail riders will have to pay a lot more attention to the transit maps and the destination signs on the trains. To make room for the future  Silver Line trains  and to accommodate the increased number of people heading toward the eastern side of downtown D.C.,  Metro  will  modify service on the Blue, Yellow and Orange lines  during rush hours. Orange Line trains will be sent to Largo Town Center as well as Landover. Some Blue Line trains will be redesignated as Yellow Line trains, and they will travel between Franconia-Springfield and Greenbelt. Look for the old lines going to new places on a  revised version of the Metro map . Metro maintenance After the holiday lull, the transit authority will resume its  aggressive maintenance program  on the rail system. During the last three weekends of January, for example, some stations on the Orange, Blue and Red lines are scheduled to be closed, and Metro will shift riders to shuttle buses to get around the closings. Metro will finish off the fixes to the Foggy Bottom station entrance   by opening the stairway and installing a protective canopy, and in February, it also will begin replacement of the escalators at the south entrance to the Dupont Circle station, closing that entrance for much of 2012. Metro fare increase? All the maintenance disruptions should put riders in a swell mood to hear about potential fare increases.  Metro General Manager Richard Sarles  will propose his next budget this month. But his chief financial officer, Carol Kissal, said in December that a  fare increase would likely be part of the package . The transit staff also will look at simplifying the complex fare structure, which is based on distance traveled and time of day. I hope that will include eliminating the “peak-of-the-peak” rate for the height of rush hour. Advocates envisioned that in part as a congestion management technique, but it’s been just one more way of baffling tourists. More road work Many transportation efforts fall below the ribbon-cutting scale in grandeur but still have a big impact on daily commuting, both as work zones and as completed projects. For 2012, they will include continued lane shifts and lane narrowings for  Northwest Branch bridge rehabilitation  on the Beltway, resurfacing of the Beltway between Arena Drive and D’Arcy Road, resurfacing of I-66 between the Beltway and Route 50, the beginning of the Washington Boulevard bridge over Columbia Pike, construction on the Linton Hall Road overpass at Route 29 in Gainesville, and a “Great Streets” safety and beautification project on Minnesota Avenue in D.C. Riders, walkers, bikers There are plenty of transit and pathway projects that will benefit travelers. They include additional bus routes using the Intercounty Connector; the planned expansion of the  Capital Bikeshare rental program , adding 50 stations and 500 bikes; construction of a pedestrian bridge over the railroad tracks to the Rhode Island Avenue Metro station; construction of a pedestrian bridge between the Minnesota Avenue Metro station and Kenilworth Avenue to the Parkside community; and construction of the Anacostia Riverwalk Trail’s Kenilworth Gardens segment."
        assert ", ".join(parsed["links"]) == "http://www.washingtonpost.com/local, http://www.washingtonpost.com/local/commuting/big-changes-in-store-for-tysons-corner-travelers-with-future-hot-lanes-setup/2011/11/21/gIQAQvPezN_story.html, http://www.washingtonpost.com/2011/02/22/ABV7qSI_category.html?blogId=dr-gridlock&cat=Capital%20Beltway, http://www.washingtonpost.com/blogs/dr-gridlock/post/congestion-relief-coming-on-dulles-toll-road/2011/12/16/gIQAKWtiyO_blog.html, http://www.washingtonpost.com/2010/07/06/AB9gLQJ_linkset.html, http://www.washingtonpost.com/local/commuting/dc-areas-top-transportation-stories-of-2011/2011/12/22/gIQAk8Z4BP_story.html, http://www.washingtonpost.com/local/commuting/wilson-bridge-project-heads-for-finish/2011/06/02/AG5Ai2IH_story.html, http://www.washingtonpost.com/local/army-parking-cap-aimed-at-easing-gridlock-around-at-mark-center-in-alexandria/2011/12/15/gIQAupMiyO_story.html, http://www.washingtonpost.com/politics/two-military-medical-icons-become-one/2011/08/26/gIQAlfxFhJ_story.html, http://www.washingtonpost.com/blogs/dr-gridlock/post/new-spans-opening-at-11th-street-bridge/2011/12/15/gIQAHdvJwO_blog.html, http://www.washingtonpost.com/metrorail_to_dulles/2010/07/06/AB3YpZN_linkset.html, http://www.washingtonpost.com/dana-hedgpeth/2011/02/28/ABAxzsM_page.html, http://www.washingtonpost.com/local/metro-to-divert-blue-line-trains-along-yellow-route-in-2012/2011/10/27/gIQAmoYNNM_story.html, http://www.washingtonpost.com/local/metro-gets-to-work-on-transition-map/2011/08/29/gIQASe734J_story.html, http://www.washingtonpost.com/blogs/dr-gridlock/post/metro-announces-long-range-track-work-plan/2011/07/13/gIQAFqLfCI_blog.html, /Complete%20third%20(the%20local)%2011th%20Street%20Bridge%20and%20open%20%e2%80%9cmissing%e2%80%9d%20ramps%20between%20DC%20295%20and%20downtown., http://www.washingtonpost.com/blogs/dr-gridlock/post/sarles-to-stay-as-metro-chief/2011/08/05/gIQAD1tSxI_blog.html, http://www.washingtonpost.com/local/commuting/metro-expects-to-boost-fares/2011/12/01/gIQAA26iIO_story.html, http://www.washingtonpost.com/blogs/dr-gridlock/post/capital-beltway-gets-first-speed-cameras/2011/08/02/gIQAHUnmpI_blog.html, http://www.washingtonpost.com/local/capital-bikeshare-expansion-underway-in-dc-and-arlington/2011/11/07/gIQAtfLnwM_story.html"
        assert parsed["url"] == "https://www.washingtonpost.com/local/commuting/the-year-ahead-for-dc-regions-commuters/2011/12/27/gIQA5W5VUP_story.html"

    def test_parse_article_2013(self):
        raw = self.articles[1]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "Business Digest: Sears to spin off Lands’ End, last-minute bid to block airline merger fails"
        assert parsed["offset_first_paragraph"] == 195
        assert parsed["date"] == 1386378264000
        assert parsed["kicker"] == "Business"
        assert parsed["author"] == ""
        assert parsed["text"] == "RETAIL Sears will spin off its Lands’ End unit Sears Holdings said Friday that it will spin off its Lands’ End clothing business as a separate company by distributing stock to Sears shareholders. It’s the latest move by the struggling retailer to turn around its results as it faces wider losses and increasingly displeased investors. Sears had said in October that it was considering separating the Lands’ End and Sears Auto Center businesses from the rest of the company. It did not mention Sears Auto Center in Friday’s announcement. Belus Capital Advisors analyst Brian Sozzi said the move shows Sears was unable to get a buyer at the right price for Lands’ End and may raise questions about how much other well-known brand names Sears owns, such as Craftsman, are worth. “It makes you question the value of what Sears is sitting on,” he said. Edward Lampert, Sears’s chairman and chief executive, disclosed recently that his stake in the company has been reduced to less than 50 percent as investors pulled money out of his hedge fund. Sears continues to face losses. Last month, it reported a wider third-quarter loss as revenue declined 7 percent, to $8.27 billion. The company heavily marked down goods to move merchandise in the quarter. Lands’ End, which sells clothing and home goods on the Internet and through catalogues, began in 1963 as a sailboat  hardware- and-equipment catalogue but morphed into a clothing company by 1977. Sears bought the company in 2002.  — Associated Press  AIRLINES Bid to block merger of AMR, US Air fails A federal judge on Friday rejected a last-ditch effort by consumers and travel agents to stop American Airlines and US Airways from merging next week, a move some fear would drive ­prices up and service down and make planes more crowded. The combination of American’s parent AMR Corp. and US Airways Group would create the world’s largest carrier. Last month, the companies resolved the Justice Department’s antitrust concerns. That settlement requires the airlines to shed some landing slots and gates at several airports, including in New York and Washington. The settlement was approved last week by the bankruptcy judge overseeing AMR’s Chapter 11 case. AMR has said it hopes to complete the merger Monday. In their appeal in the U.S. District Court in Manhattan, plaintiffs led by California resident Carolyn Fjord urged that the bankruptcy court’s order be put on hold, saying they would face irreparable harm if the “anti- competitive” merger went forward. The consumers and travel agents said combining the carriers could result in fewer flights and available seats, higher fares, poorer service and lower competition, and would be hard to undo once completed. At a hearing Friday, Chief Judge Loretta Preska of the U.S. District Court in Manhattan said the plaintiffs had failed to show irreparable harm.  — Reuters  Also in Business ●  A group of 13 defendants  who had been charged in a cyber- attack on PayPal’s Web site pleaded guilty to the December 2010 incident in response to PayPal’s suspension of WikiLeaks accounts. The pleas took place in a California federal court Thursday and were announced Friday by the U.S. attorney’s office in San Francisco. After the release of classified documents by WikiLeaks, PayPal suspended its accounts so the anti-secrecy Web site could no longer receive donations. In retribution, the group Anonymous coordinated and executed denial-of- service attacks against PayPal. ●  Former Goldman Sachs Group trader  Matthew Taylor was sentenced Friday to nine months in prison and ordered pay $118 million in restitution to his former employer after he pleaded guilty to pursuing an unauthorized $8.3 billion futures trade in 2007. U.S. District Judge William Pauley in New York imposed the sentence eight months after Taylor turned himself in to federal authorities and admitted to wire fraud. Prosecutors said Taylor fabricated trades to conceal an $8.3 billion position in Standard & Poor’s 500 E-mini futures contracts, which bet on the direction that stock index would take. ●  Americans  increased their borrowing by $18.2 billion in October, to a seasonally adjusted $3.08 trillion, the Federal Reserve reported Friday. The increase was led by a $13.9 billion rise in borrowing for auto loans and student loans. Borrowing in the category that covers credit cards rose by $4.3 billion, the biggest monthly gain since May. That category of borrowing had fallen $218 million in September. ●  A federal judge  has granted ­final approval to Bank of America’s record $500 million settlement with investors who claimed they were misled by the bank’s Countrywide unit into buying risky mortgage debt. In a decision made public Friday, U.S. District Judge Mariana Pfaelzer in Los Angeles called the accord fair, reasonable and adequate. She also awarded the investors’ lawyers $85 million in fees and $2.98 million for expenses. Investors, including several public and union pension funds, had accused Countrywide of misleading them in documents about the quality of home loans underlying the securities they bought between 2005 and 2007.  — From news services"
        assert ", ".join(parsed["links"]) == ""
        assert parsed["url"] == "https://www.washingtonpost.com/business/economy/business-digest-sears-to-spin-off-lands-end-last-minute-bid-to-block-airline-merger-fails/2013/12/06/d8693d06-5eb9-11e3-95c2-13623eb2b0e1_story.html"

    def test_parse_article_2014(self):
        raw = self.articles[2]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "Hawaii election to be held Friday in precincts closed by storm"
        assert parsed["offset_first_paragraph"] == 273
        assert parsed["date"] == 1407851892000
        assert parsed["kicker"] == "Post Politics"
        assert parsed["author"] == "Sean Sullivan"
        assert parsed["text"] == '''Updated at 9:50 a.m. Tuesday  Hawaii elections officials have scheduled a Friday election in two Big Island precincts that were closed due to storm damage during Saturday's primary. This means the deadlocked Democratic race for U.S. Senate is likely to be decided that day. The decision to hold an in-person election marks a shift from previously announced plans to have voters in those precincts return absentee ballots in the coming weeks. Damage from Tropical Storm Iselle prevented voters from casting ballots on Saturday. The vote in the Puna precincts are expected to decide the outcome of the hotly contested Democratic primary for U.S. Senate, which remains too close to call. Sen. Brian Schatz  leads  Rep. Colleen Hanabusa by 1,635 votes. The Associated Press has not called the race. “We’re about to send a letter to all voters stating the time, place, some of the provisions of the polling place. The counties are also going to post signs on the highways,” Chief Election Officer Scott Nago  told KHON . The two outstanding precincts are home to about 8,200 voters, some of whom voted by absentee ballot in advance of the primary. Hanabusa  appears to face a tough climb  toward closing the gap between her and Schatz. In a statement, Hanabusa spokesman Peter Boylan called the state's decision "disappointing" and said the campaign is reviewing its legal options. "A lot of voters in those two precincts are without power and water and many of the roads are blocked with debris, isolating large pockets of the community," he said. "It is unrealistic to think people struggling to find basic necessities and get out of their homes will have the ability to go to the polls Friday." Schatz's campaign did not immediately respond to a request for comment.'''
        assert ", ".join(parsed["links"]) == "http://www.washingtonpost.com/blogs/post-politics/wp/2014/08/10/hawaii-governor-falls-to-democratic-primary-challenger/, http://khon2.com/2014/08/11/walk-in-vote-to-be-held-friday-for-2-big-island-precincts/, http://www.washingtonpost.com/blogs/the-fix/wp/2014/08/11/colleen-hanabusa-hasnt-lost-the-hawaii-senate-primary-yet-but-she-almost-certainly-will/"
        assert parsed["url"] == "https://www.washingtonpost.com/news/post-politics/wp/2014/08/11/hawaii-democratic-senate-primary-could-be-resolved-by-friday-or-saturday/"

    def test_parse_article_2015(self):
        raw = self.articles[3]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "Japan and South Korea argue over a chocolate-covered pretzel stick"
        assert parsed["offset_first_paragraph"] == 224
        assert parsed["date"] == 1447248899000
        assert parsed["kicker"] == "WorldViews"
        assert parsed["author"] == "Anna Fifield"
        assert parsed["text"] == '''South Korea and Japan fight over a group of guano-covered rocky islets in the sea between them. They argue over their recollections of history in the first half of the 20th century, when Japan colonized the Korean Peninsula. Now, they're arguing over pretzel sticks dipped in chocolate. Today, Nov. 11, is Pepero Day. Well, it is if you’re in South Korea. If you’re in Japan, it’s  Pocky Day.  Not letting any commercial opportunity slip by, food companies in both countries have turned 11/11 — a date marked by four long lines — into a celebration of their respective long and skinny snacks. And, as with so many issues between them, there’s a dispute about who “owns” the day, something that has escalated as each company promotes its product in the other country — and farther afield in Asia. In Japan, Pepero has gained popularity thanks to the “Korean wave” of films and music. In South Korea, Japanese products are still widely considered more high-end than homemade ones. The fight says much about the similarities, as well as the antagonism, between the two countries. Some background: There is no doubt that the treat originated in Japan. Ezaki Glico, the Japanese confectionery company,  brought out Pocky in 1966,  promoting it as a “snack with a handle,” as the chocolate doesn’t extend all the way to the bottom. The name Pocky represents the snapping sound made while eating it — pokkin pokkin — to the Japanese ear, the company’s Web site says. Lotte Confectionery, a South Korean food company, started making a strikingly similar product — while denying it had copied Pocky  —  called Pepero in 1983. But the dispute arises over who commercialized 11/11 first. Lotte presents this as an organic event that began in the mid-1990s when middle-school girls in South Korea started exchanging Peperos, promising each other that they would become as skinny as the sticks. It took off. Now, sales from the Pepero Day season, which extends from September to November, account for as much as half of Lotte’s annual Pepero sales, according to Yonhap News. This year, consumers are expected to spend up to 20 percent more for the day, as it falls one day before the College Scholastic Ability Test, Hankook Ilbo reports. In Japan, Nov. 11 is officially known as Pocky and Pretz Day, including its plainer, non-candied pretzel cousin Pretz, since about 1999. Glico has emphasized the originality of its Pocky snack, particularly through social media marketing. In 2012, it set a Guinness World Record for the most-tweeted brand name in a 24-hour period. There’s a Pocky-themed  dance contest with J Soul Brothers , a music group, and  a festival in Osaka , in collaboration with Tsutenkaku, the landmark tower. As Nov. 11 dawned in Asia, both companies launched social media campaigns. People, both ordinary and famous, posted photos of themselves celebrating with the stick snacks. Seems everyone can agree on one thing: Chocolate-covered pretzels are delicious. And that, if nothing else, is a marketing triumph.  Read more:   Leaders of Japan and South Korea agree to keep talking -- that's a breakthrough   Japanese cartoonist is slammed for portraying men as house husbands   Today's coverage from our correspondents around the world'''
        assert ", ".join(parsed["links"]) == "http://pocky.glico.com/1111/, http://pocky.glico.com/history/, http://cp.pocky.jp/sharehappi/, http://www.pocky.jp/event/tsutenkaku2015/index.html, https://www.washingtonpost.com/world/leaders-of-japan-and-s-korea-agree-to-keep-talking--thats-a-breakthrough/2015/11/02/906101c1-b003-4b00-9605-b99b080b93ae_story.html, http://www.washingtonpost.com/news/worldviews/wp/2015/10/27/japanese-cartoonist-is-slammed-for-portraying-men-as-house-husbands/, https://www.washingtonpost.com/world/"
        assert parsed["url"] == "https://www.washingtonpost.com/news/worldviews/wp/2015/11/11/japan-and-south-korea-argue-over-a-chocolate-covered-pretzel-stick/"

    def test_parse_article_2016(self):
        raw = self.articles[4]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "Why party bosses can’t contain Trump"
        assert parsed["offset_first_paragraph"] == 85
        assert parsed["date"] == 1454106524000
        assert parsed["kicker"] == "Opinions"
        assert parsed["author"] == "H. W. Brands"
        assert parsed["text"] == '''H. W. Brands is the author of “Reagan: The Life” and other books of American history.   If Donald Trump wins the Republican nomination for president over the strenuous efforts of party elites to derail him, he ought to send a note of thanks to Geoffrey Cowan. Almost 50 years ago, Cowan led a campaign among the Democrats to strengthen the system of primary elections and reduce the power of party bosses. The campaign succeeded, giving the Democrats George McGovern in 1972 and spilling over into the GOP in time for Ronald Reagan to demonstrate in 1980 that primary voters were less worried about his age than the party pros. Cowan is currently president of the Annenberg Foundation Trust and a professor at the University of Southern California. In “Let the People Rule,” he examines the origins of the primary system during the Progressive era of the early 20th century. Progressives decried the debilitation of democracy at the hands of corporate moguls and political bosses. They tackled the moguls with antitrust laws and business regulations; they circumvented the bosses with reforms that made democracy more direct. These included the popular election of senators, the initiative and the referendum, and primary elections. In 1910 Oregon adopted a measure establishing the nation’s first presidential primaries. The idea had sufficient appeal that several other states scheduled primaries ahead of the 1912 presidential election. In that year Theodore Roosevelt, the former president, challenged William Howard Taft, the incumbent, for the Republican nomination. Taft controlled the party machinery, but Roosevelt hoped to leverage his personal popularity against the president. Roosevelt had been tepid on the subject of primaries, but finding his way back to the White House blocked by the Taft regulars, he had a conversion experience. The primary, he suddenly proclaimed, was essential to good government. “I believe that the majority of the plain people of the United States will, day in and day out, make fewer mistakes in governing themselves than any smaller class or body of men, no matter what their training, will make in trying to govern them,” he told a packed house at Carnegie Hall in March 1912. Cowan tells his story with great verve. He relates the experience of Roosevelt supporters in Oklahoma, who included veterans of TR’s Rough Rider regiment from the Spanish-American War. “Our fellows put up a great fight lasting all day and until four in the morning,” one of them wrote to Roosevelt. “One man dropped dead and two or three were carried out unconscious. The state chairman, Harris, a Taft man, was told if he tried to put over any crooked deals from the chair that he wouldn’t get out of the hall alive. Feelings ran so high that gun-play was expected. Indeed, I am told that one of Roosevelt’s men stood behind Harris with his hand on his gun ready for an emergency.” Cowan’s tale is packed with such vignettes, portraying a pre-radio, pre-television, pre-Internet time when politics was conducted face to face. Cowan’s Roosevelt slashes the air with his right hand while making key points; he clips his words with a speaking style that one contemporary likened to biting off tenpenny nails. This Roosevelt was also capable of turning on a dime without conceding that he had gone anywhere but straight. “Let the people rule,” Roosevelt declared, but he didn’t mean all the people all the time. When it served his purposes to include African Americans, he was more than happy to accept their votes. But when the black vote worked against him — when, having won most of the primaries but lost the Republican nomination to Taft, he became the nominee of the Progressive Party and feared frightening off Southern whites — he took pains to exclude them. “I believe the great majority of the negroes in the South are wholly unfit for suffrage,” he declared in what was either a moment of candor or a moment of expedience. Cowan surmises it was a bit of both. He nonetheless admires Roosevelt for opening presidential politics to greater participation. Cowan acknowledges the drawbacks of primaries. “The primary process has produced a new class of political leaders and insiders who are not necessarily representative of the general public or even of the party,” he writes. Primaries tempt or compel candidates to appease the most extreme elements in their parties. And they amplify the influence of money. “Primary campaigns have become so costly that candidates are forced to spend much of their time raising money and trying to win the hearts of a few very large donors.” On balance, though, Cowan applauds what Roosevelt wrought. If not for primaries, he suggests, neither John Kennedy in 1960 nor Barack Obama in 2008 would have become president. And Reagan might have lost his third and doubtless final try to gain the White House. Cowan tips his cap to TR in concluding that primaries have indeed, as Roosevelt promised they would, “given the people the right to rule.” And given them the right to make Trump the Republican nominee. Should such an outcome occur, those frustrated GOP regulars might wish TR had wrought less well. Let the People Rule Theodore Roosevelt and the Birth of the Presidential Primary By Geoffrey Cowan Norton. 404 pp. $27.95'''
        assert ", ".join(parsed["links"]) == ""
        assert parsed["url"] == "https://www.washingtonpost.com/opinions/why-party-bosses-cant-contain-trump/2016/01/29/c53b2f88-b64a-11e5-9388-466021d971de_story.html"

    def test_parse_article_2017(self):
        raw = self.articles[5]
        parsed = self.parser.parse_article(raw)
        assert parsed["title"] == "A bold paint color and neutral furniture turns a dreary room into a cozy retreat"
        assert parsed["offset_first_paragraph"] == 565
        assert parsed["date"] == 1490835610000
        assert parsed["kicker"] == "Home & Garden"
        assert parsed["author"] == "Mari-Jane Williams"
        assert parsed["text"] == '''THE CHALLENGE Lori Stillman has lovingly restored her historic home in Manassas but desperately needs a new look for the 13-by-15-foot family room. Her 13-year-old daughter has declared it a “mishmash of grandma furniture,” she says, so she knows it’s time to tackle this space. She can move the computer to another room and is looking for suggestions on making the space a warm, comfortable spot for watching TV and gathering, while maintaining the flow of traffic. She gravitates toward a transitional style and wants well-made pieces that will hold up over time. THE PROPOSED SOLUTION Designer Michelle Borden suggests a bold paint color for the small, bright space and keeps the placement of the sofa and television while adding versatile seating. She blends traditional pieces with some subtle mid-century modern accents as a nod to the tastes of Stillman and her teenage daughter. BORDEN’S SUGGESTIONS A small space with lots of windows will not feel dark. Try a dramatic paint color, such as  Sherwin-Williams’s Porpoise . Use light furniture and curtains to balance it out. Have light furniture pieces treated to resist soil and stains. Try a white cellular shade paired with wool-blend curtains for insulation without sacrificing natural light. Move the desk and chair to the bay window alcove, and hang a piece of art there to create a focal point. Choose a variety of seating options, including armchairs and stools, for a versatile space that will accommodate guests when needed. Opt for neutral, rather than colorful, accents that use texture and variety to create visual interest. The hide area rug, velvet armchair and white lacquer media console help accomplish this. Hang the TV on the wall, about four to five inches above the console, to free up the surface for decorative elements. Borden, with Perceptions Interiors (202-330-5619,  perceptions interiors.com ), is based in the District. SPLURGE OR SAVE SPLURGE: Solange desk ($2,745,  mgbwhome.com ), left. SAVE: Mid-century desk in white ($599,  westelm.com ). SPLURGE: Lawson chest ($3,230,  mgbwhome.com ), left. SAVE: Mirella mirrored 54-inch TV stand ($700,  pier1.com ). SHOPPING GUIDE  Furniture:  Essex sofa in natural with casters ($2,199,  crateandbarrel.com ); clear acrylic Zella accent table ($230,  worldmarket.com ); tufted ottoman in natural ($449,  cb2.com ); Angela collection multi dining chair ($220,  pier1.com  ); velvet Holloway armchair in light gray ($1,298,  anthropologie.com ).  Accessories:  Library accent lamp in antique brass with English-barrel linen shade in white, size C ($204,  restorationhardware. com ); Explosion chandelier ($799,  potterybarn. com );  acrylic tray  ($44 for 14-by-18-inch) and  floating wood floor mirror in white lacquer  ($399), both from  westelm.com ; “Symbolism” abstract wall art ($399,  wisteria.com ); Monroe ­ 8-by-10-foot hide rug ($1,199,  cb2.com ); Daniel tree stool ($150,  allmodern.com );  three-fourths-inch light-filtering single-cell cellular shades in ivory  (from $237) and  pinch-pleat wool-blend custom drapery panels in snow  (from $409), both from  theshadestore.com .  More from  House Calls :   Follow us on Pinterest.   See answers to frequently asked questions about House Calls here   Tell us about your own design challenge here   See past room makeovers by local designers here'''
        assert ", ".join(parsed["links"]) == "https://www.sherwin-williams.com/homeowners/color/find-and-explore-colors/paint-colors-by-family/SW7047-porpoise#/7047/?s=coordinatingColors&p=PS0, http://www.perceptionsinteriors.com, http://www.mgbwhome.com/solange-desk/11238-DSK.html?cgid=Desks, http://www.westelm.com/products/mid-century-desk-white-h208/?pkey=coffice-desks&&coffice-desks, http://www.mgbwhome.com/lawson-chest/10481-CNT.html?cgid=Media-and-Entertainment#start=5, http://www.pier1.com/mirella-mirrored-54%22-tv-stand/3005269.html, http://www.crateandbarrel.com/essex-sofa-with-casters/s308043, http://www.worldmarket.com/product/clear+acrylic+zella+accent+table.do?&from=fn, https://www.cb2.com/tufted-natural-ottoman/s241155, http://www.pier1.com/angela-multi-dining-chair/2996522.html?cgid=dining-chairs#nav=tile&icid=cat_furniture-subcat_dining_furniture-subcat_tile_dining_chairs&start=1&sz=30&showAll=166, https://www.anthropologie.com/shop/velvet-holloway-armchair?color=006&quantity=1&size=ALL&type=REGULAR, https://www.restorationhardware.com/catalog/product/product.jsp?productId=prod1157156&categoryId=cat10160019, http://www.potterybarn.com/products/explosion-chandelier/?pkey=cchandeliers&&cchandeliers, http://www.westelm.com/products/acrylic-trays-e707/?pkey=cstorage-trays&&cstorage-trays, http://www.westelm.com/products/floating-wood-floor-mirror-w538/?cm_src=PIPRecentView, http://www.westelm.com, http://www.wisteria.com/Abstract-Wall-Art-Symbolism-NEW/productinfo/T16338, https://www.cb2.com/monroe-hide-rug/f11855, https://www.allmodern.com/Daniel-Tree-Stool-IMX12665.html, https://www.theshadestore.com/shades/cellular-shades/custom-cellular-shades?preselected_collections[]=3-4-inch-single-cell-light-filtering, https://www.theshadestore.com/drapery/custom-drapes/pinch-pleat-drapery, http://www.theshadestore.com, http://www.washingtonpost.com/lifestyle/style, https://www.pinterest.com/washingtonpost/home-makeovers/, https://www.washingtonpost.com/lifestyle/home/frequently-asked-questions-about-house-calls/2015/04/15/5cad8baa-e1ed-11e4-81ea-0649268f729e_story.html, mailto:makeover@washpost.com, http://www.washingtonpost.com/housecalls"
        assert parsed["url"] == "https://www.washingtonpost.com/lifestyle/home/a-bold-paint-color-and-neutral-furniture-turns-a-dreary-room-into-a-cozy-retreat/2017/03/29/f1edab98-07f0-11e7-93dc-00f9bdd74ed1_story.html"
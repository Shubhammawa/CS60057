import xml.etree.ElementTree as ET
import csv

tree = ET.parse("event_extraction_nlp_project/hindi_anntated/Train/heatwave-blankets-japan-kills-14-people-over-long-weekend-2018071700276.txt.xml")
root = tree.getroot()

# open a file for writing

Hindi_data = open('Hindi_data.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(Hindi_data)
_head = []

count = 0
for member in root.findall('p'):
	casualities_arg = []
	w = []
	place_arg = []
	natural_event = []
	if count == 0:
		word = member.find('w').tag
		w.append(word)
		# PhoneNumber = member.find('PhoneNumber').tag
		# resident_head.append(PhoneNumber)
		# EmailAddress = member.find('EmailAddress').tag
		# resident_head.append(EmailAddress)
		# Address = member[3].tag
		# resident_head.append(Address)
		csvwriter.writerow(w)
		count = count + 1

	word = member.find('w').text
	w.append(word)
	# PhoneNumber = member.find('PhoneNumber').text
	# resident.append(PhoneNumber)
	# EmailAddress = member.find('EmailAddress').text
	# resident.append(EmailAddress)
	# Address = member[3][0].text
	# address_list.append(Address)
	# City = member[3][1].text
	# address_list.append(City)
	# StateCode = member[3][2].text
	# address_list.append(StateCode)
	# PostalCode = member[3][3].text
	# address_list.append(PostalCode)
	# resident.append(address_list)
	csvwriter.writerow(w)
Hindi_data.close()
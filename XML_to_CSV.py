<State>
<Resident Id="100">
<Name>Sample Name</Name>
	<PhoneNumber>1234567891</PhoneNumber>
	<EmailAddress>sample_name@example.com</EmailAddress>
	<Address>
		<StreetLine1>Street Line1</StreetLine1>
		<City>City Name</City>
		<StateCode>AE</StateCode>
		<PostalCode>12345</PostalCode>
	</Address>
</Resident>
<Resident Id="101">
	<Name>Sample Name1</Name>
	<PhoneNumber>1234567891</PhoneNumber>
	<EmailAddress>sample_name1@example.com</EmailAddress>
	<Address>
		<StreetLine1>Current Address</StreetLine1>
		<City>Los Angeles</City>
		<StateCode>CA</StateCode>
		<PostalCode>56666</PostalCode>
	</Address>
</Resident>
.
.
.
.
</State>

import xml.etree.ElementTree as ET
import csv

tree = ET.parse("resident_data.xml")
root = tree.getroot()

# open a file for writing

Resident_data = open('/tmp/ResidentData.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(Resident_data)
resident_head = []

count = 0
for member in root.findall('Resident'):
	resident = []
	address_list = []
	if count == 0:
		name = member.find('Name').tag
		resident_head.append(name)
		PhoneNumber = member.find('PhoneNumber').tag
		resident_head.append(PhoneNumber)
		EmailAddress = member.find('EmailAddress').tag
		resident_head.append(EmailAddress)
		Address = member[3].tag
		resident_head.append(Address)
		csvwriter.writerow(resident_head)
		count = count + 1

	name = member.find('Name').text
	resident.append(name)
	PhoneNumber = member.find('PhoneNumber').text
	resident.append(PhoneNumber)
	EmailAddress = member.find('EmailAddress').text
	resident.append(EmailAddress)
	Address = member[3][0].text
	address_list.append(Address)
	City = member[3][1].text
	address_list.append(City)
	StateCode = member[3][2].text
	address_list.append(StateCode)
	PostalCode = member[3][3].text
	address_list.append(PostalCode)
	resident.append(address_list)
	csvwriter.writerow(resident)
Resident_data.close()
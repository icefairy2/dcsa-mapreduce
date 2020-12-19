import sys
from datetime import datetime

from mrjob.job import MRJob
from mrjob.step import MRStep

import csv


class TopTenCustomers(MRJob):

    def mapper_revenue_by_year_customer(self, _, line):
        """
        This mapper yields each year (extracted from the invoice date in the fifth column), the customer ids (seventh column)
        and the revenue for each one (quantity * price)
        :param _: None
        :param line: one line from the input file
        :return: ((year, customer_id), revenue)
        """

        # For input file type CSV, skip first line and split other lines by commas
        if line != "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country":
            attributes = list(csv.reader([line]))[0]

            # Columns: Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country
            quantity = float(attributes[3])
            invoice_date = attributes[4]
            price = float(attributes[5])
            customer_id = attributes[6]

            # Take into account only customer ids different than NULL
            if customer_id != '':
                year = datetime.strptime(invoice_date, '%m/%d/%Y %H:%M:%S').year

                revenue = price * quantity
                yield (year, customer_id), revenue

    def combiner_sum_revenue(self, year_customer_pair, revenue):
        """
        This combiner sums the revenues we've computed so far by key
        :param year_customer_pair: (year, customer_id)
        :param revenue: price * quantity
        :return: ((year, customer_id), sum of revenue)
        """
        yield (year_customer_pair, sum(revenue))

    def reducer_sum_revenue(self, year_customer_pair, revenue):
        """
        This reducer sends all (total revenue, (year, customer)) constructs to the next step
        :param year_customer_pair: pair obtained from the combiner
        :param revenue: the total revenue of the key from the result of the combiner
        :return: (None, (sum(revenue), year_customer_pair))
        """
        yield None, (sum(revenue), year_customer_pair)

    def mapper_customer_revenues_by_year(self, _, revenues_year_customer_pair):
        """
        This mapper remaps the result of the previous step for our final reducer
        :param _: None
        :param revenues_year_customer_pair: (revenue, (year, customer))
        :return: (genre, (count, keyword))
        """
        revenue = revenues_year_customer_pair[0]
        year = revenues_year_customer_pair[1][0]
        customer = revenues_year_customer_pair[1][1]
        yield year, (revenue, customer)

    def reducer_find_top_ten_customers(self, year, revenue_customer_pair):
        """
        This reducer gets the top 10 client with the higher total revenue per year
        :param year
        :param revenue_customer_pair: each item of revenue_customer_pair is (revenue, customer),
        :return: (key=revenue, value=customer) 15 times
        """

        # Sort by value of the revenue
        sorted_revenue_customer_pairs = sorted(revenue_customer_pair, key=lambda x: x[0], reverse=True)
        for i in range(10):
            yield year, sorted_revenue_customer_pairs[i]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_revenue_by_year_customer,
                   combiner=self.combiner_sum_revenue,
                   reducer=self.reducer_sum_revenue),
            MRStep(mapper=self.mapper_customer_revenues_by_year,
                   reducer=self.reducer_find_top_ten_customers)
        ]


if __name__ == '__main__':
    TopTenCustomers.run()

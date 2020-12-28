import sys
from datetime import datetime

from mrjob.job import MRJob
from mrjob.step import MRStep

import csv


class TopTenCustomers(MRJob):

    def mapper_revenue_customer(self, _, line):
        """
        This mapper yields the customer ids (seventh column) and the revenue for each one (quantity * price)
        :param _: None
        :param line: one line from the input file
        :return: (customer_id, revenue)
        """

        # For input file type CSV, skip first line and split other lines by commas
        if line != "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country":
            attributes = list(csv.reader([line]))[0]

            # Columns: Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country
            quantity = float(attributes[3])
            price = float(attributes[5])
            customer_id = attributes[6]

            # Take into account only customer ids different than NULL
            if customer_id != '':
                revenue = price * quantity
                yield customer_id, revenue

    def combiner_sum_revenue(self, customer_id, revenue):
        """
        This combiner sums the revenues we've computed so far by key
        :param customer_id
        :param revenue: price * quantity
        :return: (customer_id, sum of revenue)
        """
        yield customer_id, sum(revenue)

    def reducer_sum_revenue(self, customer_id, revenue):
        """
        This reducer sends all (total revenue, customer) constructs to the next step
        :param customer_id
        :param revenue: the total revenue of the key from the result of the combiner
        :return: (None, (sum(revenue), customer_id))
        """
        yield None, (sum(revenue), customer_id)

    def reducer_find_top_ten_customers(self, _, revenue_customer_pair):
        """
        This reducer gets the top 10 client with the higher total revenue for each year
        :param _: discard the key; it is just None
        :param revenue_customer_pair: each item of revenue_customer_pair is (revenue, customer)
        :return: (key=revenue, value=customer) 10 times
        """

        # Sort by value of the revenue
        sorted_revenue_customer_pairs = sorted(revenue_customer_pair, key=lambda x: x[0], reverse=True)
        for i in range(10):
            yield sorted_revenue_customer_pairs[i]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_revenue_customer,
                   combiner=self.combiner_sum_revenue,
                   reducer=self.reducer_sum_revenue),
            MRStep(reducer=self.reducer_find_top_ten_customers)
        ]


if __name__ == '__main__':
    TopTenCustomers.run()

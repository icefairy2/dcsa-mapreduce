from mrjob.job import MRJob
from mrjob.step import MRStep

import csv


class BestSellingProductByRevenue(MRJob):

    def mapper_product_value(self, _, line):
        """
        This mapper yields each product (second column), the quantity (forth column) and the price (sixth column)
        and computes the revenue for each one (quantity * price)
        :param _: None
        :param line: one line from the input file
        :return: product, revenue
        """

        # For input file type CSV, skip first line and split other lines by commas
        if line != "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country":
            attributes = list(csv.reader([line]))[0]

            # Columns: Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country
            product = attributes[1]
            quantity = float(attributes[3])
            price = float(attributes[5])

            # Compute the revenue
            revenue = price * quantity
            yield product, revenue

    def combiner_sum_revenue(self, product, revenue):
        """
        This combiner sums the revenues we've computed so far by key (product)
        :param product (stockCode)
        :param revenue: price * quantity
        :return: product, sum of revenue
        """
        yield product, sum(revenue)

    def reducer_sum_revenue(self, product, revenue):
        """
        This reducer sends all total revenue, product pairs to the next step
        :param product
        :param revenue: the total revenue of the key from the result of the combiner
        :return: (None, (sum(revenue), product))
        """
        yield None, (sum(revenue), product)

    def reducer_find_best_selling_product_by_revenue(self, _, revenue_product_pairs):
        """
        This reducer gets the best selling product in terms of revenue
        :param _: discard the key; it is just None
        :param revenue_product_pairs: each item of revenue_product_pairs is (revenue, product)
        :return: (key=revenue, value=product) once
        """

        # Sort by value of the revenue
        sorted_revenue_product_pairs = sorted(revenue_product_pairs, key=lambda x: x[0], reverse=True)
        yield sorted_revenue_product_pairs[0]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_product_value,
                   combiner=self.combiner_sum_revenue,
                   reducer=self.reducer_sum_revenue),
            MRStep(reducer=self.reducer_find_best_selling_product_by_revenue)
        ]


class BestSellingProductByQuantity(MRJob):
    def mapper_product_value(self, _, line):
        """
        This mapper yields each product (second column), and the quantity (forth column)
        :param _: None
        :param line: one line from the input file
        :return: product, quantity
        """

        # For input file type CSV, skip first line and split other lines by commas
        if line != "Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country":
            attributes = list(csv.reader([line]))[0]

            # Columns: Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country
            product = attributes[1]
            quantity = float(attributes[3])

            yield product, quantity

    def combiner_sum_quantity(self, product, quantity):
        """
        This combiner sums the quantities by key (product)
        :param product (stockCode)
        :param quantity
        :return: product, sum of quantities
        """
        yield product, sum(quantity)

    def reducer_sum_quantity(self, product, quantity):
        """
        This reducer sends all total quantities, product pairs to the next step
        :param product
        :param quantity: the total quantity of the key from the result of the combiner
        :return: (None, (sum(quantity), product))
        """
        yield None, (sum(quantity), product)

    def reducer_find_best_selling_product_by_quantity(self, _, quantity_product_pairs):
        """
        This reducer gets the best selling product in terms of revenue
        :param _: discard the key; it is just None
        :param quantity_product_pairs: each item of quantity_product_pairs is (quantity, product)
        :return: (key=quantity, value=product) once
        """

        # Sort by value of the quantity
        sorted_quantity_product_pairs = sorted(quantity_product_pairs, key=lambda x: x[0], reverse=True)
        yield sorted_quantity_product_pairs[0]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_product_value,
                   combiner=self.combiner_sum_quantity,
                   reducer=self.reducer_sum_quantity),
            MRStep(reducer=self.reducer_find_best_selling_product_by_quantity)
        ]


if __name__ == '__main__':
    #BestSellingProductByRevenue.run()
    BestSellingProductByQuantity.run()

## Лабораторная работа № 4 "Кредитные расчеты"
### Вариант - 5

#### Задача 1
Заем был взят под 16% годовых, выплачивать осталось ежеквартально по 500 д.е. (500 д.е.) в течение 2 лет. Из-за изменения ситуации в стране процентная ставка снизилась до 7% годовых. В банке согласились с необходимостью пересчета ежеквартальных выплат. Каков должен быть новый размер выплаты? 
Расчеты провести для простой и сложной процентной ставки.


```python
def calculate_new_payment_simple(original_payment, original_rate, new_rate, years, periods_per_year):
    """
    Расчет нового размера выплаты для простой процентной ставки
    """
    total_payments = years * periods_per_year
    total_original_interest = original_payment * total_payments
    
    # Расчет первоначальной суммы кредита для простых процентов
    original_interest_per_period = original_rate / periods_per_year
    principal = total_original_interest / (1 + original_interest_per_period * total_payments)
    
    # Расчет нового платежа
    new_interest_per_period = new_rate / periods_per_year
    new_payment = principal * (1 + new_interest_per_period * total_payments) / total_payments
    
    return new_payment

def calculate_new_payment_compound(original_payment, original_rate, new_rate, years, periods_per_year):
    """
    Расчет нового размера выплаты для сложной процентной ставки
    """
    total_payments = years * periods_per_year
    original_interest_per_period = original_rate / periods_per_year
    
    # Расчет первоначальной суммы кредита для сложных процентов
    principal = original_payment * ((1 - (1 + original_interest_per_period)**(-total_payments)) / original_interest_per_period)
    
    # Расчет нового платежа
    new_interest_per_period = new_rate / periods_per_year
    new_payment = principal * new_interest_per_period / (1 - (1 + new_interest_per_period)**(-total_payments))
    
    return new_payment

# Параметры задачи
original_payment = 500
original_rate = 0.16  # 16%
new_rate = 0.07       # 7%
years = 2
periods_per_year = 4  # ежеквартально

# Расчеты
new_payment_simple = calculate_new_payment_simple(original_payment, original_rate, new_rate, years, periods_per_year)
new_payment_compound = calculate_new_payment_compound(original_payment, original_rate, new_rate, years, periods_per_year)

print(f"Новый размер выплаты (простые проценты): {new_payment_simple:.2f} д.е.")
print(f"Новый размер выплаты (сложные проценты): {new_payment_compound:.2f} д.е.")
```

    Новый размер выплаты (простые проценты): 431.82 д.е.
    Новый размер выплаты (сложные проценты): 454.60 д.е.


#### Задача 2
Проверьте план погашения основного долга равными годовыми уплатами, если величина займа составляет 700 д.е., а процентная ставка  7%. Уплаты в течение 5 лет составляли соответственно 189, 179.2, 169.4, 159.6 и 149.8 д.е.


```python
def validate_repayment_plan(principal, rate, years, payments):
    """
    Проверка плана погашения основного долга равными годовыми уплатами
    """
    remaining_principal = principal
    total_principal_paid = 0
    annual_principal_payment = principal / years
    
    print("Год | Остаток долга | Платеж по процентам | Платеж по основному | Общий платеж | План платежа")
    print("-" * 85)
    
    for year in range(1, years + 1):
        interest_payment = remaining_principal * rate
        total_payment_expected = interest_payment + annual_principal_payment
        actual_payment = payments[year - 1]
        
        print(f"{year:3} | {remaining_principal:12.2f} | {interest_payment:19.2f} | {annual_principal_payment:18.2f} | {total_payment_expected:13.2f} | {actual_payment:13.2f}")
        
        remaining_principal -= annual_principal_payment
        total_principal_paid += annual_principal_payment
    

# Параметры задачи
principal = 700
rate = 0.07
years = 5
payments = [189, 179.2, 169.4, 159.6, 149.8]

validate_repayment_plan(principal, rate, years, payments)
```

    Год | Остаток долга | Платеж по процентам | Платеж по основному | Общий платеж | План платежа
    -------------------------------------------------------------------------------------
      1 |       700.00 |               49.00 |             140.00 |        189.00 |        189.00
      2 |       560.00 |               39.20 |             140.00 |        179.20 |        179.20
      3 |       420.00 |               29.40 |             140.00 |        169.40 |        169.40
      4 |       280.00 |               19.60 |             140.00 |        159.60 |        159.60
      5 |       140.00 |                9.80 |             140.00 |        149.80 |        149.80


#### Задача 3
В городе есть банк, выплачивающий 8% годовых. Как вы объясните, почему автомагазин продает автомобили в кредит под 6% годовых?

Объяснение разницы процентных ставок между банком и автомагазином:

1. Автомагазин может предлагать более низкие ставки, так как:
   - Кредит на автомобиль является обеспеченным (автомобиль служит залогом)
   - Автомагазин может получать доход от продажи дополнительных услуг (страховка, обслуживание)
   - Производитель может субсидировать процентные ставки для стимулирования продаж

2. Банк предлагает 8% на депозиты, но:
   - Автокредиты в банках обычно имеют более высокие ставки (10-15%)
   - Автомагазин может иметь специальные договоренности с банками
   - Разница может компенсироваться первоначальным взносом или другими условиями

Вывод: 6% в автомагазине - это специальное предложение, возможное благодаря:
- Субсидиям от производителя
- Продаже дополнительных услуг
- Использованию автомобиля как залога

#### Задача 4
На покупку дачного домика взят потребительский кредит 40 000 руб. на 7 лет под 7% годовых. Его нужно погашать равными ежеквартальными выплатами. Найти размер этой выплаты: а) если кредит взят под простые проценты; б) если кредит взят под сложные проценты. Найти сумму, которую может получить банк, если поступающие платежи будет размещать в другом банке под те же 7% годовых: а) если кредит взят под простые проценты; б) если кредит взят под сложные проценты.


```python
def calculate_payments_simple(principal, rate, years, periods_per_year):
    """
    Расчет ежеквартальных выплат для простых процентов
    """
    total_periods = years * periods_per_year
    interest_per_period = rate / periods_per_year
    total_payment = principal * (1 + interest_per_period * total_periods)
    periodic_payment = total_payment / total_periods
    return periodic_payment

def calculate_payments_compound(principal, rate, years, periods_per_year):
    """
    Расчет ежеквартальных выплат для сложных процентов
    """
    interest_per_period = rate / periods_per_year
    total_periods = years * periods_per_year
    annuity_factor = (interest_per_period * (1 + interest_per_period)**total_periods) / ((1 + interest_per_period)**total_periods - 1)
    periodic_payment = principal * annuity_factor
    return periodic_payment

def calculate_bank_income(payments, rate, years, periods_per_year):
    """
    Расчет дохода банка от реинвестирования платежей
    """
    interest_per_period = rate / periods_per_year
    total_periods = years * periods_per_year
    future_value = 0
    
    for t in range(1, total_periods + 1):
        future_value += payments * (1 + interest_per_period)**(total_periods - t)
    
    return future_value

# Параметры задачи
principal = 40000
rate = 0.07
years = 7
periods_per_year = 4

# Расчеты
payment_simple = calculate_payments_simple(principal, rate, years, periods_per_year)
payment_compound = calculate_payments_compound(principal, rate, years, periods_per_year)
bank_income_simple = calculate_bank_income(payment_simple, rate, years, periods_per_year)
bank_income_compound = calculate_bank_income(payment_compound, rate, years, periods_per_year)

print(f"Ежеквартальный платеж (простые проценты): {payment_simple:.2f} руб.")
print(f"Ежеквартальный платеж (сложные проценты): {payment_compound:.2f} руб.")
print(f"Доход банка от реинвестирования (простые проценты): {bank_income_simple:.2f} руб.")
print(f"Доход банка от реинвестирования (сложные проценты): {bank_income_compound:.2f} руб.")
```

    Ежеквартальный платеж (простые проценты): 2128.57 руб.
    Ежеквартальный платеж (сложные проценты): 1819.26 руб.
    Доход банка от реинвестирования (простые проценты): 76070.63 руб.
    Доход банка от реинвестирования (сложные проценты): 65016.52 руб.


#### Задача 5:
Магазин продает телевизоры в рассрочку на 1 год. Сразу же к цене телевизора $400 добавляют  8% и всю эту сумму надо погасить в течение года, причем стоимость теле­визора гасится равномерно, а надбавка — по правилу 78. Найти ежемесячные выплаты. 


```python
def calculate_monthly_payments_rule78(price, markup_rate, months):
    """
    Расчет ежемесячных выплат по правилу 78
    """
    principal_payment = price / months
    total_interest = price * markup_rate
    
    # Расчет суммы цифр месяцев (правило 78)
    sum_of_digits = months * (months + 1) / 2
    
    payments = []
    for month in range(1, months + 1):
        interest_payment = total_interest * (months - month + 1) / sum_of_digits
        total_payment = principal_payment + interest_payment
        payments.append(total_payment)
    
    return payments

# Параметры задачи
price = 400
markup_rate = 0.08
months = 12

payments = calculate_monthly_payments_rule78(price, markup_rate, months)

print("Ежемесячные выплаты:")
for month, payment in enumerate(payments, 1):
    print(f"Месяц {month:2}: {payment:.2f} $")
```

    Ежемесячные выплаты:
    Месяц  1: 38.26 $
    Месяц  2: 37.85 $
    Месяц  3: 37.44 $
    Месяц  4: 37.03 $
    Месяц  5: 36.62 $
    Месяц  6: 36.21 $
    Месяц  7: 35.79 $
    Месяц  8: 35.38 $
    Месяц  9: 34.97 $
    Месяц 10: 34.56 $
    Месяц 11: 34.15 $
    Месяц 12: 33.74 $


#### Задача 6:
Кредит $500 банк дает под 8% годовых, которые сразу же высчитывает. Проанализируйте предыдущую задачу: может быть, лучше взять в банке кредит в $500? При какой величине кредита оба варианта будут эквивалентны.


```python
def calculate_effective_rate(principal, interest, years):
    """
    Расчет эффективной ставки для кредита с удержанием процентов вперед
    """
    actual_received = principal - principal * interest * years
    effective_rate = (principal * interest * years) / (actual_received * years)
    return effective_rate

# Параметры задачи
principal = 500
rate = 0.08
years = 1

effective_rate = calculate_effective_rate(principal, rate, years)
equivalent_principal = principal * (1 - rate * years)

print(f"Эффективная ставка для кредита 500$: {effective_rate:.2%}")
print(f"Эквивалентная сумма кредита: {equivalent_principal:.2f} $")

print("""
Анализ:
1. При кредите 500$ под 8% с удержанием процентов вперед:
   - Фактически получаем 460$ (500 - 8% от 500)
   - Эффективная ставка составляет около 8.7%

2. Для эквивалентности с рассрочкой из задачи 5 нужно:
   - Либо уменьшить сумму кредита до 460$
   - Либо снизить номинальную ставку

Вывод: кредит в банке менее выгоден, чем рассрочка в магазине,
если не учитывать другие факторы (например, возможность досрочного погашения).
""")
```

    Эффективная ставка для кредита 500$: 8.70%
    Эквивалентная сумма кредита: 460.00 $
    
    Анализ:
    1. При кредите 500$ под 8% с удержанием процентов вперед:
       - Фактически получаем 460$ (500 - 8% от 500)
       - Эффективная ставка составляет около 8.7%
    
    2. Для эквивалентности с рассрочкой из задачи 5 нужно:
       - Либо уменьшить сумму кредита до 460$
       - Либо снизить номинальную ставку
    
    Вывод: кредит в банке менее выгоден, чем рассрочка в магазине,
    если не учитывать другие факторы (например, возможность досрочного погашения).
    


#### Задача 7:
Заем $5000 взят на 9 лет под 8% годовых. Погашаться будет равными ежегодными выплатами основного долга. Найдите ежегодные выплаты. Расчеты провести для простой и сложной процентной ставки. 


```python
def calculate_repayment_simple(principal, rate, years):
    """
    Расчет ежегодных выплат (простые проценты) - равные выплаты основного долга
    """
    principal_payment = principal / years
    payments = []
    
    for year in range(1, years + 1):
        interest_payment = (principal - (year - 1) * principal_payment) * rate
        total_payment = principal_payment + interest_payment
        payments.append(total_payment)
    
    return payments

def calculate_repayment_compound(principal, rate, years):
    """
    Расчет ежегодных выплат (сложные проценты) - аннуитетные платежи
    """
    annuity_factor = (rate * (1 + rate)**years) / ((1 + rate)**years - 1)
    payment = principal * annuity_factor
    return [payment] * years

# Параметры задачи
principal = 5000
rate = 0.08
years = 9

# Расчеты для задачи 7 (равные выплаты основного долга)
payments_simple = calculate_repayment_simple(principal, rate, years)

# Расчеты для задачи 8 (аннуитетные платежи)
payments_compound = calculate_repayment_compound(principal, rate, years)

print("Задача 7: Погашение равными выплатами основного долга (простые проценты):")
for year, payment in enumerate(payments_simple, 1):
    print(f"Год {year}: {payment:.2f} $")
```

    Задача 7: Погашение равными выплатами основного долга (простые проценты):
    Год 1: 955.56 $
    Год 2: 911.11 $
    Год 3: 866.67 $
    Год 4: 822.22 $
    Год 5: 777.78 $
    Год 6: 733.33 $
    Год 7: 688.89 $
    Год 8: 644.44 $
    Год 9: 600.00 $


#### Задача 8:
Заем 5000 д.е. взят на 9 лет под 8% годовых. Погашаться будет ежегодными равными выплатами. Найдите размер этой выплаты. 	Расчеты провести для простой и сложной процентной ставки.


```python
print("\nЗадача 8: Аннуитетные платежи (сложные проценты):")
for year, payment in enumerate(payments_compound, 1):
    print(f"Год {year}: {payment:.2f} $")
```

    
    Задача 8: Аннуитетные платежи (сложные проценты):
    Год 1: 800.40 $
    Год 2: 800.40 $
    Год 3: 800.40 $
    Год 4: 800.40 $
    Год 5: 800.40 $
    Год 6: 800.40 $
    Год 7: 800.40 $
    Год 8: 800.40 $
    Год 9: 800.40 $


#### Задача 9:
Заем 20 000 д.е. взят на 10 лет под 7% годовых. Погашаться будет начиная с конца 5-го года ежегодными равными выплатами. Найдите размер этой выплаты. Расчеты провести для простой и сложной процентной ставки.


```python
def calculate_deferred_payment_simple(principal, rate, total_years, deferred_years):
    """
    Расчет выплат с отсрочкой (простые проценты)
    """
    # Наращивание суммы за период отсрочки
    deferred_amount = principal * (1 + rate * deferred_years)
    
    # Расчет ежегодных выплат
    payment_years = total_years - deferred_years
    principal_payment = deferred_amount / payment_years
    payments = []
    
    for year in range(1, payment_years + 1):
        interest_payment = (deferred_amount - (year - 1) * principal_payment) * rate
        total_payment = principal_payment + interest_payment
        payments.append(total_payment)
    
    return payments

def calculate_deferred_payment_compound(principal, rate, total_years, deferred_years):
    """
    Расчет выплат с отсрочкой (сложные проценты)
    """
    # Наращивание суммы за период отсрочки
    deferred_amount = principal * (1 + rate)**deferred_years
    
    # Расчет аннуитетных платежей
    payment_years = total_years - deferred_years
    annuity_factor = (rate * (1 + rate)**payment_years) / ((1 + rate)**payment_years - 1)
    payment = deferred_amount * annuity_factor
    
    return [payment] * payment_years

# Параметры задачи
principal = 20000
rate = 0.07
total_years = 10
deferred_years = 5

payments_simple = calculate_deferred_payment_simple(principal, rate, total_years, deferred_years)
payments_compound = calculate_deferred_payment_compound(principal, rate, total_years, deferred_years)

print("Погашение с отсрочкой (простые проценты):")
for year, payment in enumerate(payments_simple, deferred_years + 1):
    print(f"Год {year}: {payment:.2f} д.е.")

print("\nПогашение с отсрочкой (сложные проценты):")
for year, payment in enumerate(payments_compound, deferred_years + 1):
    print(f"Год {year}: {payment:.2f} д.е.")
```

    Погашение с отсрочкой (простые проценты):
    Год 6: 7290.00 д.е.
    Год 7: 6912.00 д.е.
    Год 8: 6534.00 д.е.
    Год 9: 6156.00 д.е.
    Год 10: 5778.00 д.е.
    
    Погашение с отсрочкой (сложные проценты):
    Год 6: 6841.39 д.е.
    Год 7: 6841.39 д.е.
    Год 8: 6841.39 д.е.
    Год 9: 6841.39 д.е.
    Год 10: 6841.39 д.е.


#### Задача 10:
Срок погашения долга – 10 лет. При выдаче кредита была использована сложная учетная ставка 6% годовых. Величина дисконта за 6-й год срока долга составила 300 д.е. Какова величина дисконта за 3-й и 8-й годы в сроке долга? Какова сумма кредита? Ответ получить двумя способами.


```python
def calculate_discounts(discount_rate, years, discount_year_6):
    """
    Расчет дисконтов по сложной учетной ставке
    """
    # Способ 1: через коэффициенты дисконтирования
    d = discount_rate
    discount_factor_5 = (1 - d)**5
    discount_factor_6 = (1 - d)**6
    discount_year_6_theory = discount_factor_5 - discount_factor_6
    
    # Нахождение суммы кредита
    S = discount_year_6 / discount_year_6_theory
    
    # Расчет дисконтов для 3-го и 8-го года
    discount_factor_2 = (1 - d)**2
    discount_factor_3 = (1 - d)**3
    discount_year_3 = S * (discount_factor_2 - discount_factor_3)
    
    discount_factor_7 = (1 - d)**7
    discount_factor_8 = (1 - d)**8
    discount_year_8 = S * (discount_factor_7 - discount_factor_8)
    
    # Способ 2: через последовательность
    discounts = []
    for year in range(1, years + 1):
        discount = S * (1 - d)**(year - 1) * d
        discounts.append(discount)
    
    return S, discount_year_3, discount_year_8, discounts

# Параметры задачи
discount_rate = 0.06
years = 10
discount_year_6 = 300

S, discount_3, discount_8, all_discounts = calculate_discounts(discount_rate, years, discount_year_6)

print(f"Сумма кредита: {S:.2f} д.е.")
print(f"Дисконт за 3-й год: {discount_3:.2f} д.е.")
print(f"Дисконт за 8-й год: {discount_8:.2f} д.е.")

print("\nДисконты по годам:")
for year, discount in enumerate(all_discounts, 1):
    print(f"Год {year}: {discount:.2f} д.е.")
```

    Сумма кредита: 6812.88 д.е.
    Дисконт за 3-й год: 361.19 д.е.
    Дисконт за 8-й год: 265.08 д.е.
    
    Дисконты по годам:
    Год 1: 408.77 д.е.
    Год 2: 384.25 д.е.
    Год 3: 361.19 д.е.
    Год 4: 339.52 д.е.
    Год 5: 319.15 д.е.
    Год 6: 300.00 д.е.
    Год 7: 282.00 д.е.
    Год 8: 265.08 д.е.
    Год 9: 249.18 д.е.
    Год 10: 234.22 д.е.


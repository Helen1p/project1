import os
import json
import random
from typing import List, Tuple

# 目前可能还不够多
# which, what: pure answer / choice A.B.C.
# yes/no 有点诡异???
q_template = {
   'segment': ['Generate the distortion-region segmentation map.'],
#   整体失真level比较
  'all': 
  ['which is the most degraded region in all the regions? ',
  'Which region exhibits the highest level of degradation among all the regions? ',
  'Which region shows the most severe degradation across all the regions? ', 

  'which is the least degraded region in all the regions? ', 
  'Which region is least affected by degradation compared to others regions? ',
  'Which region has experienced the minimal level of degradation in all the regions? ',
  'Which region exhibits the lowest degree of degradation compared to all the other regions? '],
#   单一失真level比较：最大和最小 注意：不比较统一区域的
  # 'Which region has the most noticeable distortion of [Gaussian noise]',
  # 'Which region is most affected by [distortion]',
  # 'What region shows the highest impact from [distortion]',
  # 'What region is slightly impacted by [distortion]',
  'single': 
  ['Which region shows the highest level of ',
  'Which region shows the most severe ',
  'What region has the highest amount of ',

  'What region exhibits the least amount of ',
  'What region has the minimal level of ',
  'Which region exhibits the lowest degree of '],
#   失真order
  'order': 
  ['Which region follows the distortion addition sequence of ',
  'Which region incorporates distortion in the order of ',

  'Which region add  first, followed by , and then  in distortion addition process',

  'Which region add [] last',
  'Which region add [] first']
}
q_template_choice = ['A. region ', 'B. region ', 'C. region ']
a_template = ['region ']

def os_walk(input_img, input_json, out_path, p, question: Tuple[int, int, int]):
	# question: 'all', 'single', 'order'
	# p: 生成choice回答的概率为p
	img_list = os.listdir(input_img)
	for i in img_list:
		dict_gen = grounding(name=i, input_json=input_json, p=p, question=question)
		with open(os.path.join(out_path, i.split('.')[0]+'.json'),'w') as ff:
			json.dump(dict_gen, ff)


def grounding(name, input_json, p, question: Tuple[int, int, int]):
	
	def gen_choice(region, exist_qu):
		# if random.random()<p:
		q_template_choice = ['A. region ', 'B. region ', 'C. region ']
		rand_choice_abc=random.sample(range(3), 2)
		rand_choice=random.sample(list(set(range(len(annotations['annotations'])))-set([region])), 2)
		q_template_choice[rand_choice_abc[0]]+=str(rand_choice[0])
		q_template_choice[rand_choice_abc[1]]+=str(rand_choice[1])
		answer_index = list(set([0, 1, 2]).difference(set(rand_choice_abc)))[0]
		q_template_choice[answer_index]+=str(region)
		q_template_choice2str=', '.join(q_template_choice)
		exist_qu+=q_template_choice2str
		# A. region 2.
		# answer = q_template_choice[answer_index]+'.'
		# A.
		answer = q_template_choice[answer_index].split('.')[0]+'.'
		return exist_qu, answer


	with open(os.path.join(input_json, name.split('.')[0]+'_info.json')) as f:
		annotations = json.load(f)
		all_answer=[]
		# dis_count={}
		dis_region_id={}
		dis_region_level={}

		dict_gen={name: {'all': [], 'single': [], 'order': []}}
	
		for m in range(len(annotations['annotations'])):
			level = annotations['annotations'][m]['distortion_level']
			all_answer.append(sum(level))

			distortion = annotations['annotations'][m]['distortion']
			for d in distortion:
				# if d in dis_count.keys():
				# 	dis_count[d]+=1
				# else:
				# 	dis_count[d]=1

				if d in dis_region_id.keys():
					dis_region_id[d].append(m)
				else:
					dis_region_id[d]=[m]

				if d in dis_region_level.keys():
					dis_region_level[d].append(level[distortion.index(d)])
				else:
					dis_region_level[d]=[level[distortion.index(d)]]
			dis_region_level_out = {k:v for k, v in dis_region_level.items() if len(set(v))>1}
			# dis_count_out={ k:v for k,v in dis_count.items() if v>1 }	
			
			order_choice=random.sample(range(len(q_template['order'])), question[2])
			if len(distortion)>1:
				for i in order_choice:
					if i <2:
						qu=q_template['order'][i]
						for di in range(len(distortion)):
							qu+=distortion[di]
							if di==len(distortion)-1:
								qu+='? '
							else:
								qu+=', '
					elif i==2:
					# 'Which region add  first, followed by , and then  in distortion addition process',
						if len(distortion)==2:
							qu='Which region add '+distortion[0]+' first, followed by '+distortion[1]+' in distortion addition process? '
						elif len(distortion)==3:
							qu='Which region add '+distortion[0]+' first, followed by '+distortion[1]+', and then '+distortion[2]+' in distortion addition process? '
					elif i==3:
					# 'Which region add [] last'
						qu='Which region add '+ distortion[-1]+' last? '
					elif i==4:
					# 'Which region add [] first'
						qu='Which region add '+ distortion[0]+' first? '

					if random.random()<p:
						qu, order_answer = gen_choice(region=m, exist_qu=qu)
					else:
						order_answer=a_template[0]+str(m)+'.'

					dict_gen[name]['order'].append(qu)
					dict_gen[name]['order'].append(order_answer)

		most_num=all_answer.index(max(all_answer))
		least_num=all_answer.index(min(all_answer))


		all_choice=random.sample(range(len(q_template['all'])), question[0])
		for i in all_choice:
			if i < 3:
				num = most_num
			else:
				num = least_num
			# answer = a_template[0]+str(num)+'.'
			qu = q_template['all'][i]

			if random.random()<p:
				qu, answer = gen_choice(region=num, exist_qu=qu)
			else:
				answer = a_template[0]+str(num)+'.'
		
			dict_gen[name]['all'].append(qu)
			dict_gen[name]['all'].append(answer)

		single_choice=random.sample(range(len(q_template['single'])), question[1])

		if dis_region_level_out is None:
			pass
		else:
			if len(single_choice)>len(dis_region_level_out):
				single_choice=single_choice[:len(dis_region_level_out)]

			dis_region_level_out=sorted(dis_region_level_out.items(), key=lambda x: len(x[1]), reverse=True)

			for i in single_choice:
				target_dis=dis_region_level_out[single_choice.index(i)][0]
				qu = q_template['single'][i]+target_dis+'? '

				# 去掉level相同的
				if len(set(dis_region_level[target_dis]))>1:
					# 最大
					if i < 3:
						region_index=dis_region_level[target_dis].index(max(dis_region_level[target_dis]))
					else:
						region_index=dis_region_level[target_dis].index(min(dis_region_level[target_dis]))
					region_id=dis_region_id[target_dis][region_index]
					# region_id ???????
					if random.random()<p:
						qu, answer = gen_choice(region=region_id, exist_qu=qu)
					else:
						answer = a_template[0]+str(region_id)+'.'

					dict_gen[name]['single'].append(qu)
					dict_gen[name]['single'].append(answer)

	return dict_gen


# def wrong():


if __name__ == '__main__':
    os_walk(input_img=r'/root/autodl-tmp/example_new/DIV2K_output1', input_json=r'/root/autodl-tmp/example_new/json',
             out_path=r'/root/autodl-tmp/example_new/grounding', p=0.3, question=[1, 2, 1])
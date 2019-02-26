%
% superclass for all fields of view
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-02-03
%
classdef field_of_view

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
%         N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2	% number of dimensions (1)
        str_name = { 'default' }                                                            % name of the field of view

	end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field_of_view( N_dimensions )
            
            % return if no argument
            if nargin == 0
                return;
            end

            % construct objects
            N_objects = numel( N_dimensions );
            objects = repmat( objects, size( N_dimensions ) );

            % set independent properties
%             for index_object = 1:N_objects
%                 objects( index_object ).N_dimensions = N_dimensions( index_object );
%             end

        end % function objects = field_of_view( N_dimensions )

    end

end % classdef field_of_view

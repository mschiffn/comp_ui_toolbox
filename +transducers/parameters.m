%
% superclass for all transducer array parameters
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-03-25
%
classdef parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_elements_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 1 ]	% numbers of elements along each coordinate axis (1)
        str_model = 'Default Array'         % model name
        str_vendor = 'Default Corporation'	% vendor name

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters( N_elements_axis, str_model, str_vendor )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for N_elements_axis
            if ~iscell( N_elements_axis )
                N_elements_axis = { N_elements_axis };
            end

            % ensure cell array for str_model
            if ~iscell( str_model )
                str_model = { str_model };
            end

            % ensure cell array for str_name
            if ~iscell( str_vendor )
                str_vendor = { str_vendor };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_elements_axis, str_model, str_vendor );

            %--------------------------------------------------------------
            % 2.) create transducer array parameters
            %--------------------------------------------------------------
            objects = repmat( objects, size( N_elements_axis ) );

            % set independent properties
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_elements_axis = N_elements_axis{ index_object };
                objects( index_object ).str_model = str_model{ index_object };
                objects( index_object ).str_vendor = str_vendor{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = parameters( N_elements_axis, str_name )

	end % methods

end % classdef parameters

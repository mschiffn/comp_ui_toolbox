%
% superclass for all transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-20
% modified: 2019-02-18
%
classdef array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2              % number of dimensions (1)
        N_elements_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 1 ]	% numbers of elements along each coordinate axis (1)
        str_model = 'Default Array'         % model name
        str_vendor = 'Default Corporation'	% vendor name

        % dependent properties
        N_elements ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 128              % total number of elements (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array( N_dimensions, parameters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure positive integers
            if ~all( N_dimensions(:) >= 1 && abs( N_dimensions(:) - floor( N_dimensions(:) ) ) < eps )
                errorStruct.message     = 'N_dimensions must be positive integers!';
                errorStruct.identifier	= 'array:NoPositiveIntegers';
                error( errorStruct );
            end

            % ensure class transducers.parameters
            if ~isa( parameters, 'transducers.parameters' )
                errorStruct.message     = 'parameters must be transducers.parameters!';
                errorStruct.identifier	= 'array:NoParameters';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_dimensions, parameters );
            % assertion: N_dimensions and parameters have equal sizes

            %--------------------------------------------------------------
            % 2.) create transducer arrays
            %--------------------------------------------------------------
            % create objects
            objects = repmat( objects, size( N_dimensions ) );

            % set independent and dependent properties
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_dimensions = N_dimensions( index_object );
                objects( index_object ).N_elements_axis = parameters( index_object ).N_elements_axis( 1:objects( index_object ).N_dimensions );
                objects( index_object ).str_model = parameters( index_object ).str_model;
                objects( index_object ).str_vendor = parameters( index_object ).str_vendor;

                % dependent properties
                objects( index_object ).N_elements = prod( objects( index_object ).N_elements_axis, 2 );

            end % for index_object = 1:numel( objects )

        end

	end % methods

end % classdef array

%
% superclass for all random number generator settings
%
% author: Martin F. Schiffner
% date: 2019-01-27
% modified: 2019-01-28
%
classdef setting_rng

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        seed ( 1, 1 ) double { mustBeInteger, mustBeNonnegative }	% seed of the random number generator
        str_name ( 1, : ) char { mustBeMember( str_name, {'twister', 'simdTwister', 'combRecursive', 'multFibonacci'} ) } = 'twister'	% name of the random number generator
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rng( seeds, strs_name )

            % return if no argument
            if nargin == 0
                return;
            end

            % ensure cell array
            if ~iscell( strs_name ) || ~all( size( seeds ) == size( strs_name ) )
                errorStruct.message     = 'strs_name must be a cell array and match the size of seeds!';
                errorStruct.identifier	= 'setting_rng:NoCellOrSizeMismatch';
                error( errorStruct );
            end
            % assertion: strs_name is a cell array, all arguments have equal sizes

            % construct column vector of objects
            N_objects = numel( seeds );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects

                objects( index_object ).seed = seeds( index_object );
                objects( index_object ).str_name = strs_name{ index_object };
            end

            % reshape to sizes of the arguments
            objects = reshape( objects, size( seeds ) );

        end % function objects = setting_rng( seeds, strs_name )

	end % methods
    
end % classdef setting_rng

%
% superclass for all random number generator (RNG) settings
%
% author: Martin F. Schiffner
% date: 2019-01-27
% modified: 2019-05-10
%
classdef setting_rng

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        seed ( 1, 1 ) double { mustBeInteger, mustBeNonnegative }	% seed of the random number generator
        str_name ( 1, : ) char { mustBeMember( str_name, {'twister', 'simdTwister', 'combRecursive', 'multFibonacci'} ) } = 'twister'	% name of the random number generator

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rng( seeds, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure nonempty strs_name
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                strs_name = varargin{ 1 };
            else
                strs_name = repmat( { 'twister' }, size( seeds ) );
            end

            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( seeds, strs_name );

            %--------------------------------------------------------------
            % 2.) create RNG settings
            %--------------------------------------------------------------
            % repeat default RNG settings
            objects = repmat( objects, size( seeds ) );

            % iterate RNG settings
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).seed = seeds( index_object );
                objects( index_object ).str_name = strs_name{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = setting_rng( seeds, strs_name )

	end % methods
    
end % classdef setting_rng

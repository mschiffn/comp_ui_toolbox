%
% superclass for all window function settings
%
% author: Martin F. Schiffner
% date: 2019-05-24
% modified: 2019-06-21
%
classdef setting_window

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        handle ( 1, 1 ) function_handle { mustBeNonempty } = @tukeywin	% handle for window function
        parameters = { 0.1 }	% parameters (winopt value or sampling descriptor)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_window( handles, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure cell array for handles
            if ~iscell( handles )
                handles = { handles };
            end

            % ensure nonempty parameters
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                parameters = varargin{ 1 };
            else
                parameters = cell( size( handles ) );
                for index_object = 1:numel( handles )
                    parameters{ index_object } = [];
                end
            end

            % ensure cell array for parameters
            if ~iscell( parameters )
                parameters = { parameters };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( handles, parameters );

            %--------------------------------------------------------------
            % 2.) create window function settings
            %--------------------------------------------------------------
            % repeat default window function settings
            objects = repmat( objects, size( handles ) );

            % iterate window function settings
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).handle = handles{ index_object };
                objects( index_object ).parameters = parameters( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = setting_window( handles, varargin )

	end % methods
    
end % classdef setting_window

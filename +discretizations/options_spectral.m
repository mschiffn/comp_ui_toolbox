%
% superclass for all spectral discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-20
% modified: 2019-07-18
%
classdef options_spectral

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        method ( 1, 1 ) discretizations.options_spectral_method { mustBeNonempty } = discretizations.options_spectral_method.sequence	% spectral discretization method
        interval_hull_t ( :, 1 ) math.interval	% custom recording time interval

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spectral( methods, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure nonempty intervals_hull_t
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                intervals_hull_t = varargin{ 1 };
            else
                intervals_hull_t = repmat( [], size( methods ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( methods, intervals_hull_t );

            %--------------------------------------------------------------
            % 2.) create spectral discretization options
            %--------------------------------------------------------------
            % repeat default spectral discretization options
            objects = repmat( objects, size( methods ) );

            % iterate spectral discretization options
            for index_object = 1:numel( methods )

                % set independent properties
                objects( index_object ).method = methods( index_object );
                objects( index_object ).interval_hull_t = intervals_hull_t( index_object );

            end % for index_object = 1:numel( spatial )

        end % function objects = options_spectral( methods, varargin )

    end % methods

end % classdef options_spectral
